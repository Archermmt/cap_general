"""Genesis drone hover agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cap_general.core.agent import BaseAgent, BaseAgentConfig

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class DroneAgentConfig(BaseAgentConfig):
    """Configuration for DroneAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_drone_hover_robot"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    horizon: int = 1000


@BaseAgent.register()
class DroneAgent(BaseAgent):
    """Agent that evaluates Genesis drone hover policies."""

    name = "Genesis Drone Agent"
    config_cls = DroneAgentConfig

    def __init__(self, config: DroneAgentConfig, logger: Logger):
        self._policy_name = config.policy
        self.horizon = int(config.horizon)
        super().__init__(config=config, logger=logger)

    @classmethod
    def agent_type(cls) -> str:
        return "drone_agent"

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

    def follow_target(self, target_pos: list[float] | tuple[float, float, float], max_steps: int | None = None) -> dict[str, Any]:
        """Run the drone policy to fly to a fixed target position."""
        steps = int(max_steps or self.horizon)
        env = self._robot.example_env
        if env is None:
            return {
                "steps": 0,
                "target_pos": list(target_pos),
                "obs": self._robot.get_observation(self.step_dir),
                "mock": True,
            }

        self._robot.set_target_position(target_pos)
        executed_steps = self._run_policy_steps(env=env, steps=steps)
        return {"steps": executed_steps, "target_pos": list(target_pos)}

    def hover(self, time_s: float) -> dict[str, Any]:
        """Keep the drone hovering at its current position for time_s seconds."""
        duration = max(float(time_s), 0.0)
        env = self._robot.example_env
        if env is None:
            return {
                "duration": duration,
                "steps": 0,
                "obs": self._robot.get_observation(self.step_dir),
                "mock": True,
            }

        if not self._robot.target_locked():
            self._robot.hold_current_position()
        steps = int(round(duration / max(float(self._robot.dt), 1e-6)))
        executed_steps = self._run_policy_steps(env=env, steps=steps)
        return {"duration": duration, "steps": executed_steps}

    def _run_policy_steps(self, *, env: Any, steps: int) -> int:
        obs = self._robot.policy_obs
        executed_steps = 0
        for _ in range(max(int(steps), 0)):
            action = self._run_policy(self._policy_name, env=env, obs=obs)
            obs, _reward, terminated, truncated, _info = self._robot.step(action)
            executed_steps += 1
            if terminated or truncated:
                break
            obs = self._robot.policy_obs
        return executed_steps
