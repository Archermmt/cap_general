"""Genesis grasp manipulation agent."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cap_general.core.agent import BaseAgent, BaseAgentConfig

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class GraspAgentConfig(BaseAgentConfig):
    """Configuration for GraspAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_grasp"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    rl_policy: str = "runner"
    bc_policy: str = "bc"
    stage: str = "rl"
    horizon: int = 1000
    run_demo_after_episode: bool = True


@BaseAgent.register()
class GraspAgent(BaseAgent):
    """Agent that evaluates Genesis Franka grasp policies."""

    name = "Genesis Grasp Agent"
    config_cls = GraspAgentConfig

    def __init__(self, config: GraspAgentConfig, logger: Logger):
        self._rl_policy_name = config.rl_policy
        self._bc_policy_name = config.bc_policy
        self._stage = config.stage
        self._run_demo_after_episode = bool(config.run_demo_after_episode)
        super().__init__(config=config, logger=logger)

    @classmethod
    def agent_type(cls) -> str:
        return "grasp"

    def _execute_rules(self) -> str:
        return (
            "The Genesis grasp agent evaluates RL or BC policies in a robot-controlled "
            "Franka grasp scene. Use grasp_episode(stage='rl'|'bc', max_steps=...). "
            "Do not create Genesis scenes, robots, cameras, or policies in generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"grasp_episode": self.grasp_episode}

    def grasp_episode(self, stage: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
        """Run one Genesis grasp episode with an RL or BC policy."""
        current_stage = stage or self._stage
        steps = int(max_steps or self._config.horizon)
        env = self._robot.example_env
        if env is None:
            return {
                "steps": 0,
                "stage": current_stage,
                "obs": self._robot.get_observation(self.step_dir),
                "mock": True,
            }

        obs = self._robot.policy_obs
        for _ in range(steps):
            if current_stage == "rl":
                action = self._run_policy(self._rl_policy_name, obs=obs)
            elif current_stage == "bc":
                rgb_obs = self._robot.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                action = self._run_policy(
                    self._bc_policy_name,
                    env=self._robot,
                    rgb_obs=rgb_obs,
                    ee_pose=ee_pose,
                )
            else:
                raise ValueError(f"Unknown grasp stage: {current_stage!r}")
            obs, _reward, terminated, truncated, _info = self._robot.step(action)
            if terminated or truncated:
                break
            obs = self._robot.policy_obs

        if self._run_demo_after_episode:
            self._robot.grasp_and_lift_demo()
        return {"stage": current_stage}

    def _options_doc(self, method_name: str) -> str:
        """Return grasp-specific options docs for supported methods."""
        if method_name != "train":
            return super()._options_doc(method_name)
        return (
            "seed: Random seed metadata for the training run (default 1).\n"
            "train_cfg: Dict of overrides for the RL or BC training config.\n"
            "record_epoch: Training-epoch interval for summary history samples (default 50).\n"
            "summary_interval: Backward-compatible alias for record_epoch."
        )

    @staticmethod
    def _to_scalar(value: Any) -> int | float | str | bool | None:
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "mean"):
            try:
                value = value.mean()
            except Exception:
                pass
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (bool, int, float, str)):
            return value
        return None

    def _mean_value(self, values: Any) -> float | None:
        if values is None:
            return None
        if isinstance(values, (list, tuple)):
            numeric = [self._to_scalar(value) for value in values]
            numeric = [float(value) for value in numeric if isinstance(value, (int, float))]
            if not numeric:
                return None
            return sum(numeric) / len(numeric)
        value = self._to_scalar(values)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _summary(self, *, stage: str, interval: int) -> dict[str, Any]:
        return {
            "stage": stage,
            "sampling_interval": max(int(interval), 1),
            "history": [],
            "latest": None,
            "best": None,
            "available_metrics": [],
            "notes": [],
        }

    def _merge_summary(
        self,
        summary: dict[str, Any],
        point: dict[str, Any],
        *,
        best_metric: str,
        best_mode: str,
    ) -> None:
        clean = {key: self._to_scalar(value) for key, value in point.items()}
        clean = {key: value for key, value in clean.items() if value is not None}
        if not clean:
            return
        summary["latest"] = dict(clean)
        summary["available_metrics"] = sorted({*summary["available_metrics"], *clean})

        best_value = clean.get(best_metric)
        current = summary.get("best")
        if not isinstance(best_value, (int, float)):
            return
        if current is not None and isinstance(current.get("value"), (int, float)):
            if best_mode == "max" and float(best_value) <= float(current["value"]):
                return
            if best_mode == "min" and float(best_value) >= float(current["value"]):
                return
        summary["best"] = {
            "by": best_metric,
            "mode": best_mode,
            "iteration": clean.get("iteration"),
            "total_steps": clean.get("total_steps"),
            "value": float(best_value),
        }

    def _episode_metrics(self, ep_extras: list[dict[str, Any]]) -> dict[str, Any]:
        aggregated: dict[str, list[float]] = {}
        for ep_info in ep_extras:
            if not isinstance(ep_info, dict):
                continue
            for key, value in ep_info.items():
                scalar = self._mean_value(value)
                if scalar is None:
                    continue
                metric_name = f"mean_episode_{str(key).replace('/', '_')}"
                aggregated.setdefault(metric_name, []).append(float(scalar))
        return {key: sum(values) / len(values) for key, values in aggregated.items() if values}

    def _capture_rl_summary(
        self,
        runner: Any,
        *,
        interval: int,
        num_learning_iterations: int,
        init_at_random_ep_len: bool,
    ) -> dict[str, Any]:
        summary = self._summary(stage="rl", interval=interval)
        step_interval = (
            int(runner.cfg["num_steps_per_env"]) * int(runner.env.num_envs) * int(getattr(runner, "gpu_world_size", 1))
        )
        summary["notes"].append("RL metrics sampled from runner logger once per training iteration")

        next_sample_iteration = summary["sampling_interval"]
        original_log = runner.logger.log

        def wrapped_log(*args: Any, **kwargs: Any) -> Any:
            nonlocal next_sample_iteration
            extras = self._episode_metrics(list(getattr(runner.logger, "ep_extras", [])))
            result = original_log(*args, **kwargs)
            collect_time = float(kwargs.get("collect_time", 0.0))
            learn_time = float(kwargs.get("learn_time", 0.0))
            iteration_time = collect_time + learn_time
            done_iterations = int(kwargs.get("it", 0)) + 1 - int(kwargs.get("start_it", 0))
            remaining_iterations = int(kwargs.get("total_it", 0)) - int(kwargs.get("start_it", 0)) - done_iterations
            point = {
                "stage": "rl",
                "iteration": int(kwargs.get("it", 0)) + 1,
                "total_steps": int(getattr(runner.logger, "tot_timesteps", 0)),
                "steps_per_second": int(step_interval / iteration_time) if iteration_time > 0 else None,
                "collection_time": collect_time,
                "learning_time": learn_time,
                "iteration_time": iteration_time,
                "time_elapsed_seconds": float(getattr(runner.logger, "tot_time", 0.0)),
                "eta_seconds": (
                    float(getattr(runner.logger, "tot_time", 0.0)) / done_iterations * max(remaining_iterations, 0)
                    if done_iterations > 0
                    else None
                ),
                "mean_action_std": self._mean_value(kwargs.get("action_std")),
                "mean_reward": self._mean_value(getattr(runner.logger, "rewbuffer", None)),
                "mean_episode_length": self._mean_value(getattr(runner.logger, "lenbuffer", None)),
            }
            for key, value in (kwargs.get("loss_dict", {}) or {}).items():
                point[f"mean_{key}_loss"] = self._mean_value(value)
            point.update(extras)
            self._merge_summary(summary, point, best_metric="mean_episode_rew_keypoints", best_mode="max")
            iteration = point["iteration"]
            if isinstance(iteration, int) and iteration >= next_sample_iteration:
                summary["history"].append(dict(summary["latest"]))
                while iteration >= next_sample_iteration:
                    next_sample_iteration += summary["sampling_interval"]
            return result

        runner.logger.log = wrapped_log
        try:
            runner.learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            runner.logger.log = original_log
        if summary["latest"] is not None and (
            not summary["history"] or summary["history"][-1].get("iteration") != summary["latest"].get("iteration")
        ):
            summary["history"].append(dict(summary["latest"]))
        return summary

    def _capture_bc_summary(
        self,
        runner: Any,
        behavior_cloning_module: Any,
        *,
        interval: int,
        num_learning_iterations: int,
        log_dir: str,
    ) -> dict[str, Any]:
        summary = self._summary(stage="bc", interval=interval)
        num_envs = int(getattr(runner._env, "num_envs", 1))
        num_steps_per_env = int(getattr(runner, "_num_steps_per_env", 1))
        summary["notes"].append("BC metrics sampled from tensorboard writes during training iterations")

        metric_names = {
            "loss/action_loss": "action_loss",
            "loss/pose_loss": "pose_loss",
            "loss/total_loss": "total_loss",
            "lr": "learning_rate",
            "buffer_size": "buffer_size",
            "speed/forward": "forward_time",
            "speed/backward": "backward_time",
            "speed/fps": "fps",
            "reward/mean": "mean_reward",
        }
        next_sample_iteration = summary["sampling_interval"]
        points_by_iteration: dict[int, dict[str, Any]] = {}
        original_summary_writer = behavior_cloning_module.SummaryWriter

        class SummaryWriterProxy:
            def __init__(self, *args: Any, **kwargs: Any):
                self._inner = original_summary_writer(*args, **kwargs)

            def add_scalar(
                self,
                tag: str,
                scalar_value: Any,
                global_step: int | None = None,
                *args: Any,
                **kwargs: Any,
            ):
                result = self._inner.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
                metric_name = metric_names.get(tag)
                if metric_name is not None and global_step is not None:
                    iteration = int(global_step) + 1
                    total_steps = iteration * num_steps_per_env * num_envs
                    point = points_by_iteration.setdefault(
                        iteration,
                        {"stage": "bc", "iteration": iteration, "total_steps": total_steps},
                    )
                    point[metric_name] = GraspAgent._to_scalar(scalar_value)
                return result

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        behavior_cloning_module.SummaryWriter = SummaryWriterProxy
        try:
            runner.learn(num_learning_iterations=num_learning_iterations, log_dir=log_dir)
        finally:
            behavior_cloning_module.SummaryWriter = original_summary_writer

        for iteration in sorted(points_by_iteration):
            point = points_by_iteration[iteration]
            self._merge_summary(summary, point, best_metric="total_loss", best_mode="min")
            if iteration >= next_sample_iteration:
                summary["history"].append(dict(summary["latest"]))
                while iteration >= next_sample_iteration:
                    next_sample_iteration += summary["sampling_interval"]

        if summary["latest"] is None:
            final_iteration = int(getattr(runner, "_current_iter", -1)) + 1
            point = {
                "stage": "bc",
                "iteration": final_iteration if final_iteration > 0 else num_learning_iterations,
                "total_steps": max(final_iteration, num_learning_iterations) * num_steps_per_env * num_envs,
                "mean_reward": self._mean_value(getattr(runner, "_rewbuffer", None)),
            }
            self._merge_summary(summary, point, best_metric="total_loss", best_mode="min")
            summary["notes"].append("BC runner exposed no periodic scalar writes; returning final summary only")
        if summary["latest"] is not None and (
            not summary["history"] or summary["history"][-1].get("iteration") != summary["latest"].get("iteration")
        ):
            summary["history"].append(dict(summary["latest"]))
        return summary

    @staticmethod
    def _load_policy_to_runner(policy: Any, runner: Any, env: Any) -> None:
        """Load an agent policy's current weights into a training runner."""
        source_policy = getattr(policy, "_actor", None)
        if source_policy is None:
            if policy._policy is None:
                policy._ensure_loaded(env)
            source_policy = policy._policy
        if hasattr(runner, "alg"):
            runner_policy = runner.alg.get_policy().to(env.device)
        else:
            runner_policy = runner._policy
        runner_policy.load_state_dict(source_policy.state_dict())

    def _train(self, policy: Any, epoch: int, method: str, options: dict) -> tuple[dict, dict]:
        """Train a grasp policy using RL (PPO) or BC (BehaviorCloning)."""
        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as exc:
            raise ImportError("rsl-rl-lib>=5.0.0 is required for RL training.") from exc

        stage = self._stage
        if stage not in {"rl", "bc"}:
            raise ValueError(f"Unsupported grasp stage for training: {stage!r}")

        example_root = str(self._robot._config.example_root)
        if example_root not in sys.path:
            sys.path.insert(0, example_root)

        behavior_cloning_module = importlib.import_module("behavior_cloning")
        BehaviorCloning = behavior_cloning_module.BehaviorCloning

        env = self._robot
        policy_name = policy.name
        log_dir = self.train_dir / f"{policy_name}_{stage}"
        log_dir.mkdir(parents=True, exist_ok=True)

        seed = int(options.get("seed", 1))
        record_epoch = int(options.get("record_epoch", options.get("summary_interval", 50)))

        rl_cfg: dict = {
            "algorithm": {
                "class_name": "PPO",
                "clip_param": 0.2,
                "desired_kl": 0.01,
                "entropy_coef": 0.0,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 3e-4,
                "max_grad_norm": 1.0,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "schedule": "adaptive",
                "use_clipped_value_loss": True,
                "value_loss_coef": 1.0,
            },
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": [256, 256, 128],
                "activation": "relu",
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": [256, 256, 128],
                "activation": "relu",
            },
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "num_steps_per_env": 24,
            "save_interval": 100,
            "run_name": policy_name,
            "logger": "tensorboard",
        }
        bc_cfg: dict = {
            "num_steps_per_env": 24,
            "learning_rate": 1e-3,
            "num_epochs": 5,
            "num_mini_batches": 10,
            "max_grad_norm": 1.0,
            "policy": {
                "vision_encoder": {
                    "conv_layers": [
                        {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
                        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1},
                        {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1},
                    ],
                    "pooling": "adaptive_avg",
                },
                "action_head": {"state_obs_dim": 7, "hidden_dims": [128, 128, 64]},
                "pose_head": {"hidden_dims": [64, 64]},
            },
            "buffer_size": 1000,
            "log_freq": 10,
            "save_freq": 50,
            "eval_freq": 50,
        }
        if stage == "rl":
            rl_cfg.update(options.get("train_cfg", {}))
        else:
            bc_cfg.update(options.get("train_cfg", {}))

        if stage == "bc":
            rl_runner = OnPolicyRunner(env, rl_cfg, str(log_dir), device=env.device)
            self._load_policy_to_runner(self._policies[self._rl_policy_name], rl_runner, env)
            teacher_policy = rl_runner.get_inference_policy(device=env.device)
            runner = BehaviorCloning(env, bc_cfg, teacher_policy, device=env.device)
            self._load_policy_to_runner(policy, runner, env)
            summary = self._capture_bc_summary(
                runner,
                behavior_cloning_module,
                interval=record_epoch,
                num_learning_iterations=epoch,
                log_dir=str(log_dir),
            )
            new_policy = {"state_dict": runner._policy.state_dict()}
        else:
            runner = OnPolicyRunner(env, rl_cfg, log_dir, device=env.device)
            self._load_policy_to_runner(policy, runner, env)
            summary = self._capture_rl_summary(
                runner,
                interval=record_epoch,
                num_learning_iterations=epoch,
                init_at_random_ep_len=True,
            )
            new_policy = {"state_dict": runner.alg.get_policy().state_dict()}

        return (
            {
                "policy_name": policy_name,
                "stage": stage,
                "method": method,
                "train_dir": str(log_dir),
                "epoch": epoch,
                "seed": seed,
                "summary": summary,
            },
            new_policy,
        )
