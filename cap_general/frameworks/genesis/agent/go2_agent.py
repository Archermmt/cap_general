"""Genesis GO2 locomotion agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cap_general.core.agent import BaseAgent, BaseAgentConfig
from cap_general.core.utils import tensor_mean_value, tensor_to_scalar


@dataclass
class Go2AgentConfig(BaseAgentConfig):
    """Configuration for Go2Agent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_go2"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    horizon: int = 1000


@BaseAgent.register()
class Go2Agent(BaseAgent):
    """Agent that evaluates Genesis GO2 locomotion policies."""

    agent_type = "genesis_go2"
    config_cls = Go2AgentConfig

    def init_genesis(self, gs_scene: Any) -> None:
        self._robot.init_genesis(gs_scene)

    def _execute_rules(self) -> str:
        return (
            "The Genesis GO2 agent evaluates locomotion policies in a robot-controlled scene. "
            "Use walk_forward(max_steps=..., turn_angle=0.0) to make GO2 walk forward "
            "and optionally turn by a yaw angle in radians. Use stand_still(time_s=...) "
            "to keep GO2 standing still for a duration in seconds. Do not create "
            "Genesis scenes or robots in generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"walk_forward": self.walk_forward, "stand_still": self.stand_still}

    def walk_forward(self, max_steps: int | None = None, turn_angle: float = 0.0) -> dict[str, Any]:
        """Make GO2 walk forward and smoothly turn by biasing policy actions."""
        steps = int(max_steps or self._config.horizon)
        self._robot.set_walk_command(turn_angle=0.0, steps=steps)
        self._run_policy_steps(
            steps=steps,
            after_step=lambda: self._robot.set_walk_command(turn_angle=0.0, steps=steps),
            turn_angle=float(turn_angle),
        )
        return {"steps": steps, "turn_angle": float(turn_angle)}

    def stand_still(self, time_s: float) -> dict[str, Any]:
        """Keep GO2 standing still for time_s seconds."""
        duration = max(float(time_s), 0.0)
        steps = int(round(duration / max(float(self._robot.dt), 1e-6)))
        self._robot.stop_command()
        self._run_policy_steps(steps=steps, after_step=self._robot.stop_command)
        return {"duration": duration}

    def _run_policy_steps(
        self,
        *,
        steps: int,
        after_step: Callable[[], Any],
        turn_angle: float = 0.0,
    ) -> int:
        obs = self._robot.policy_obs
        for _ in range(max(int(steps), 0)):
            action = self._run_policy(self._config.policy, inputs={"obs": obs})
            action = self._robot.apply_turn_to_action(action, turn_angle)
            obs, _reward, terminated, truncated, _info = self._robot.step(action)
            after_step()
            if terminated or truncated:
                break
            obs = self._robot.policy_obs
        return None

    def _options_doc(self, method_name: str) -> str:
        """Return GO2-specific options docs for supported methods."""
        if method_name != "train":
            return super()._options_doc(method_name)
        return (
            "seed: Random seed metadata for the training run (default 1).\n"
            "train_cfg: Dict of overrides for the RSL-RL training config.\n"
            "record_epoch: Training-epoch interval for summary history samples (default 50).\n"
            "summary_interval: Backward-compatible alias for record_epoch."
        )

    @staticmethod
    def _summary(*, interval: int) -> dict[str, Any]:
        return {
            "stage": "rl",
            "sampling_interval": max(int(interval), 1),
            "history": [],
            "latest": None,
            "best": None,
            "available_metrics": [],
            "notes": [],
        }

    @staticmethod
    def _merge_summary(summary: dict[str, Any], point: dict[str, Any]) -> None:
        clean = {key: tensor_to_scalar(value) for key, value in point.items()}
        clean = {key: value for key, value in clean.items() if value is not None}
        if not clean:
            return
        summary["latest"] = dict(clean)
        summary["available_metrics"] = sorted({*summary["available_metrics"], *clean})

        best_metric = "mean_episode_rew_tracking_lin_vel"
        best_value = clean.get(best_metric)
        current = summary.get("best")
        if not isinstance(best_value, (int, float)):
            return
        if current is not None and isinstance(current.get("value"), (int, float)):
            if float(best_value) <= float(current["value"]):
                return
        summary["best"] = {
            "by": best_metric,
            "mode": "max",
            "iteration": clean.get("iteration"),
            "total_steps": clean.get("total_steps"),
            "value": float(best_value),
        }

    @staticmethod
    def _episode_metrics(ep_extras: list[dict[str, Any]]) -> dict[str, Any]:
        aggregated: dict[str, list[float]] = {}
        for ep_info in ep_extras:
            if not isinstance(ep_info, dict):
                continue
            for key, value in ep_info.items():
                scalar = tensor_mean_value(value)
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
    ) -> dict[str, Any]:
        summary = self._summary(interval=interval)
        step_interval = (
            int(runner.cfg["num_steps_per_env"])
            * int(runner.env.num_envs)
            * int(getattr(runner, "gpu_world_size", 1))
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
                "mean_action_std": tensor_mean_value(kwargs.get("action_std")),
                "mean_reward": tensor_mean_value(getattr(runner.logger, "rewbuffer", None)),
                "mean_episode_length": tensor_mean_value(getattr(runner.logger, "lenbuffer", None)),
            }
            for key, value in (kwargs.get("loss_dict", {}) or {}).items():
                point[f"mean_{key}_loss"] = tensor_mean_value(value)
            point.update(extras)
            self._merge_summary(summary, point)
            iteration = point["iteration"]
            if isinstance(iteration, int) and iteration >= next_sample_iteration:
                summary["history"].append(dict(summary["latest"]))
                while iteration >= next_sample_iteration:
                    next_sample_iteration += summary["sampling_interval"]
            return result

        runner.logger.log = wrapped_log
        try:
            runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)
        finally:
            runner.logger.log = original_log
        if summary["latest"] is not None and (
            not summary["history"] or summary["history"][-1].get("iteration") != summary["latest"].get("iteration")
        ):
            summary["history"].append(dict(summary["latest"]))
        return summary

    @staticmethod
    def _load_model_to_runner(model: Any, runner: Any, env: Any) -> None:
        """Load model weights into an RSL-RL training runner."""
        runner.alg.get_policy().to(env.device).load_state_dict(model.state_dict())

    def _train(self, policy: Any, epoch: int, options: dict) -> tuple[dict, dict]:
        """Train the GO2 locomotion policy with RSL-RL PPO."""
        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as exc:
            raise ImportError("rsl-rl-lib>=5.0.0 is required for GO2 training.") from exc

        env = self._robot
        policy_name = policy.name
        model = policy.get_model("model")
        log_dir = self.train_dir / f"{policy_name}_rl"
        log_dir.mkdir(parents=True, exist_ok=True)

        seed = int(options.get("seed", 1))
        record_epoch = int(options.get("record_epoch", options.get("summary_interval", 50)))
        train_cfg = {
            "algorithm": {
                "class_name": "PPO",
                "clip_param": 0.2,
                "desired_kl": 0.01,
                "entropy_coef": 0.01,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 0.001,
                "max_grad_norm": 1.0,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "schedule": "adaptive",
                "use_clipped_value_loss": True,
                "value_loss_coef": 1.0,
            },
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": [512, 256, 128],
                "activation": "elu",
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": [512, 256, 128],
                "activation": "elu",
            },
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "num_steps_per_env": 24,
            "save_interval": 100,
            "run_name": policy_name,
            "logger": "tensorboard",
        }
        train_cfg.update(options.get("train_cfg", {}))

        runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)
        self._load_model_to_runner(model, runner, env)
        summary = self._capture_rl_summary(
            runner,
            interval=record_epoch,
            num_learning_iterations=epoch,
        )
        new_policy = {"state_dict": runner.alg.get_policy().state_dict()}
        return (
            {
                "policy_name": policy_name,
                "stage": "rl",
                "train_dir": str(log_dir),
                "epoch": epoch,
                "seed": seed,
                "summary": summary,
            },
            new_policy,
        )
