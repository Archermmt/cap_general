"""Genesis grasp manipulation agent."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cap_general.core.agent import BaseAgent, BaseAgentConfig

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class GraspAgentConfig(BaseAgentConfig):
    """Configuration for GraspAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_grasp_robot"})
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
        self.horizon = int(config.horizon)
        self._run_demo_after_episode = bool(config.run_demo_after_episode)
        super().__init__(config=config, logger=logger)

    @classmethod
    def agent_type(cls) -> str:
        return "grasp_agent"

    def _execute_rules(self) -> str:
        return (
            "The Genesis grasp agent evaluates RL or BC policies in a robot-controlled "
            "Franka grasp scene. Use grasp_episode(stage='rl'|'bc', max_steps=...). "
            "Do not create Genesis scenes, robots, cameras, or policies in generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"grasp_episode": self.grasp_episode, "grasp_and_lift_demo": self.grasp_and_lift_demo}

    def grasp_episode(self, stage: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
        """Run one Genesis grasp episode with an RL or BC policy."""
        current_stage = stage or self._stage
        steps = int(max_steps or self.horizon)
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
                action = self._run_policy(self._rl_policy_name, env=env, obs=obs)
            elif current_stage == "bc":
                rgb_obs = self._robot.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()
                action = self._run_policy(self._bc_policy_name, env=env, rgb_obs=rgb_obs, ee_pose=ee_pose)
            else:
                raise ValueError(f"Unknown grasp stage: {current_stage!r}")
            obs, _reward, terminated, truncated, _info = self._robot.step(action)
            if terminated or truncated:
                break
            obs = self._robot.policy_obs

        demo_ran = self.grasp_and_lift_demo() if self._run_demo_after_episode else False
        return {"stage": current_stage, "demo_ran": demo_ran}

    def grasp_and_lift_demo(self) -> bool:
        """Run the scripted grasp-and-lift demo from the underlying env."""
        return bool(self._robot.grasp_and_lift_demo())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _options_doc(self, method_name: str) -> str:
        """Return grasp-specific options docs for supported methods."""
        if method_name != "train":
            return super()._options_doc(method_name)
        return (
            "num_envs: Number of parallel environments. Must match the existing robot env (default 1).\n"
            "seed: Random seed metadata for the training run (default 1).\n"
            "train_cfg: Dict of overrides for the RL or BC training config."
        )

    def _train(self, policy_name: str, epoch: int, method: str, options: dict) -> dict:
        """Train a grasp policy using RL (PPO) or BC (BehaviorCloning)."""

        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as exc:
            raise ImportError("rsl-rl-lib>=5.0.0 is required for RL training.") from exc

        method = str(method).lower()
        if method not in {"rl", "bc", "train"}:
            raise ValueError(f"Unsupported grasp train method: {method!r}")
        method = "rl" if method == "train" else method

        env = self._robot

        log_dir = self.train_dir / f"{policy_name}_{method}"
        log_dir.mkdir(parents=True, exist_ok=True)

        seed = int(options.get("seed", 1))
        configured_num_envs = int(getattr(env, "num_envs", 1))
        num_envs = int(options.get("num_envs", configured_num_envs))
        if num_envs != configured_num_envs:
            raise ValueError(
                f"train num_envs={num_envs} does not match existing robot env num_envs={configured_num_envs}"
            )

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
        if method == "rl":
            rl_cfg.update(options.get("train_cfg", {}))
        else:
            bc_cfg.update(options.get("train_cfg", {}))

        if method == "bc":
            runner_env = env.example_env
            rl_log_dir = self.train_dir / f"{policy_name}_rl"
            if not rl_log_dir.exists():
                raise FileNotFoundError(f"RL log directory {rl_log_dir} does not exist — train RL first.")
            ckpt_files = [path for path in rl_log_dir.iterdir() if re.match(r"model_\d+\.pt", path.name)]
            if not ckpt_files:
                raise FileNotFoundError(f"No RL checkpoints found in {rl_log_dir}")
            last_ckpt = max(ckpt_files, key=lambda path: int(re.search(r"\d+", path.stem).group()))
            rl_runner = OnPolicyRunner(runner_env, rl_cfg, rl_log_dir, device=runner_env.device)
            rl_runner.load(last_ckpt)
            teacher_policy = rl_runner.get_inference_policy(device=runner_env.device)

            update_result = self._update_policy(
                self._bc_policy_name,
                env=env,
                train_cfg=bc_cfg,
                teacher_policy=teacher_policy,
                log_dir=log_dir,
                epoch=epoch,
            )
        else:
            update_result = self._update_policy(
                self._rl_policy_name,
                env=env,
                train_cfg=rl_cfg,
                log_dir=log_dir,
                epoch=epoch,
                init_at_random_ep_len=True,
            )

        return {
            "policy_name": policy_name,
            "method": method,
            "train_dir": str(log_dir),
            "epoch": epoch,
            "num_envs": configured_num_envs,
            "seed": seed,
            "update": update_result,
        }
