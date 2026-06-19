"""Genesis GO2 locomotion agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cap_general.core.agent import BaseAgent, BaseAgentConfig


@dataclass
class Go2AgentConfig(BaseAgentConfig):
    """Configuration for Go2Agent."""

    env: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_go2"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    horizon: int = 1000


@BaseAgent.register()
class Go2Agent(BaseAgent):
    """Agent that evaluates Genesis GO2 locomotion policies."""

    name = "Genesis GO2 Agent"
    config_cls = Go2AgentConfig

    def __init__(self, config: Go2AgentConfig):
        self._policy_name = config.policy
        self.horizon = int(config.horizon)
        super().__init__(config=config)

    @classmethod
    def agent_type(cls) -> str:
        return "go2_agent"

    def _execute_rules(self) -> str:
        return (
            "The Genesis GO2 agent evaluates locomotion policies in an env-controlled scene. "
            "Use walk_forward(max_steps=..., turn_angle=0.0) to make GO2 walk forward "
            "and optionally turn by a yaw angle in radians. Use stand_still(time_s=...) "
            "to keep GO2 standing still for a duration in seconds. Do not create "
            "Genesis scenes or robots in generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"walk_forward": self.walk_forward, "stand_still": self.stand_still}

    def walk_forward(self, max_steps: int | None = None, turn_angle: float = 0.0) -> dict[str, Any]:
        """Make GO2 walk forward and smoothly turn by biasing policy actions."""
        steps = int(max_steps or self.horizon)
        env = self._env.example_env
        if env is None:
            return {
                "steps": 0,
                "turn_angle": float(turn_angle),
                "obs": self._env.get_observation(self._record_dir / self.step_dir),
                "mock": True,
            }

        self._env.set_walk_command(turn_angle=0.0, steps=steps)
        self._run_policy_steps(
            env=env,
            steps=steps,
            after_step=lambda: self._env.set_walk_command(turn_angle=0.0, steps=steps),
            turn_angle=float(turn_angle),
        )
        return {"steps": steps, "turn_angle": float(turn_angle)}

    def stand_still(self, time_s: float) -> dict[str, Any]:
        """Keep GO2 standing still for time_s seconds."""
        duration = max(float(time_s), 0.0)
        env = self._env.example_env
        if env is None:
            return {
                "duration": duration,
                "steps": 0,
                "obs": self._env.get_observation(self._record_dir / self.step_dir),
                "mock": True,
            }

        steps = int(round(duration / max(float(self._env.dt), 1e-6)))
        self._env.stop_command()
        self._run_policy_steps(env=env, steps=steps, after_step=self._env.stop_command)
        return {"duration": duration}

    def _run_policy_steps(
        self,
        *,
        env: Any,
        steps: int,
        after_step: Callable[[], Any],
        turn_angle: float = 0.0,
    ) -> int:
        obs = self._env.policy_obs
        for _ in range(max(int(steps), 0)):
            action = self._run_policy(self._policy_name, env=env, obs=obs)
            action = self._env.apply_turn_to_action(action, turn_angle)
            obs, _reward, terminated, truncated, _info = self._env.step(action)
            after_step()
            if terminated or truncated:
                break
            obs = self._env.policy_obs
        return None
