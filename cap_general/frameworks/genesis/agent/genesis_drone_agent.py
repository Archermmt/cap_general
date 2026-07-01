"""Genesis drone hover agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cap_general.core.agent import BaseAgentConfig
from cap_general.frameworks.genesis.agent.genesis_base_agent import GenesisBaseAgent


@dataclass
class GenesisDroneAgentConfig(BaseAgentConfig):
    """Configuration for GenesisDroneAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_drone"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    horizon: int = 1000


@GenesisBaseAgent.register()
class GenesisDroneAgent(GenesisBaseAgent):
    """Agent that evaluates Genesis drone hover policies."""

    agent_type = "genesis_drone"
    config_cls = GenesisDroneAgentConfig

    def _execute_rules(self) -> str:
        return (
            "The Genesis drone agent evaluates hover policies in a robot-controlled scene. "
            "Use follow_target(target_pos=[x, y, z], max_steps=...) to fly to a fixed "
            "target position and hover there until the step budget is exhausted. Use "
            "hover(time_s=...) to keep the drone hovering at its current position for "
            "a duration in seconds. Do not create Genesis scenes, drones, cameras, or "
            "policies in generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"follow_target": self.follow_target, "hover": self.hover}

    def follow_target(
        self, target_pos: list[float] | tuple[float, float, float], max_steps: int | None = None
    ) -> dict[str, Any]:
        """Run the drone policy to fly to a fixed target position."""
        steps = int(max_steps or self._config.horizon)
        self._robot.set_target_position(target_pos)
        executed_steps = self._run_policy_steps(steps=steps)
        return {"steps": executed_steps, "target_pos": list(target_pos)}

    def hover(self, time_s: float) -> dict[str, Any]:
        """Keep the drone hovering at its current position for time_s seconds."""
        duration = max(float(time_s), 0.0)
        if not self._robot.lock_commands:
            self._robot.set_target_position(self._robot.base_pos)
        steps = int(round(duration / max(float(self._robot.dt), 1e-6)))
        executed_steps = self._run_policy_steps(steps=steps)
        return {"duration": duration, "steps": executed_steps}

    def _run_policy_steps(self, *, steps: int) -> int:
        obs = self._robot.policy_obs
        executed_steps = 0
        for _ in range(max(int(steps), 0)):
            action = self._run_policy(self._config.policy, inputs={"obs": obs})
            obs, _reward, terminated, truncated, _info = self._robot.step(action)
            executed_steps += 1
            if terminated or truncated:
                break
            obs = self._robot.policy_obs
        return executed_steps

    def _train(self, policy: Any, epoch: int, options: dict) -> tuple[dict, dict]:
        """Train the drone hover policy with RSL-RL PPO."""
        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as exc:
            raise ImportError("rsl-rl-lib>=5.0.0 is required for drone training.") from exc

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
                "entropy_coef": 0.004,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 0.0003,
                "max_grad_norm": 1.0,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "schedule": "adaptive",
                "use_clipped_value_loss": True,
                "value_loss_coef": 1.0,
            },
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": [128, 128],
                "activation": "tanh",
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": [128, 128],
                "activation": "tanh",
            },
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "num_steps_per_env": 100,
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
            best_metric="mean_episode_rew_target",
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
