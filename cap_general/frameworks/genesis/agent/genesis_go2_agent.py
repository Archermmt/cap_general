"""Genesis GO2 locomotion agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cap_general.core.agent import BaseAgentConfig
from cap_general.frameworks.genesis.agent.genesis_base_agent import GenesisBaseAgent


@dataclass
class GenesisGo2AgentConfig(BaseAgentConfig):
    """Configuration for GenesisGo2Agent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_go2"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    horizon: int = 1000


@GenesisBaseAgent.register()
class GenesisGo2Agent(GenesisBaseAgent):
    """Agent that evaluates Genesis GO2 locomotion policies."""

    agent_type = "genesis_go2"
    config_cls = GenesisGo2AgentConfig

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
            best_metric="mean_episode_rew_tracking_lin_vel",
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
