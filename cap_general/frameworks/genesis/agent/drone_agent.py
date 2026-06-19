"""Genesis drone hover agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cap_general.core.agent import BaseAgent, BaseAgentConfig


@dataclass
class DroneAgentConfig(BaseAgentConfig):
    """Configuration for DroneAgent."""

    env: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_drone_hover"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    horizon: int = 1000


@BaseAgent.register()
class DroneAgent(BaseAgent):
    """Agent that evaluates Genesis drone hover policies."""

    name = "Genesis Drone Agent"
    config_cls = DroneAgentConfig

    def __init__(self, config: DroneAgentConfig):
        self._policy_name = config.policy
        self.horizon = int(config.horizon)
        super().__init__(config=config)

    @classmethod
    def agent_type(cls) -> str:
        return "drone_agent"

    def _execute_rules(self) -> str:
        return (
            "The Genesis drone agent evaluates hover policies in an env-controlled scene. "
            "Use hover(max_steps=...) to let the drone hover and follow sampled target "
            "commands. Do not create Genesis scenes, drones, cameras, or policies in "
            "generated code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"hover": self.hover}

    def hover(self, max_steps: int | None = None) -> dict[str, Any]:
        """Run the drone hover policy for max_steps simulation steps."""
        steps = int(max_steps or self.horizon)
        env = self._env.example_env
        if env is None:
            return {
                "steps": 0,
                "obs": self._env.get_observation(self._record_dir / self.step_dir),
                "mock": True,
            }

        obs = self._env.policy_obs
        executed_steps = 0
        for _ in range(max(steps, 0)):
            action = self._run_policy(self._policy_name, env=env, obs=obs)
            obs, _reward, terminated, truncated, _info = self._env.step(action)
            executed_steps += 1
            if terminated or truncated:
                break
            obs = self._env.policy_obs
        return {"steps": executed_steps}
