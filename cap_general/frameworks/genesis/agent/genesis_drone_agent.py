"""Genesis drone hover agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cap_general.frameworks.genesis.agent.genesis_base_agent import GenesisBaseAgent, GenesisBaseAgentConfig


@dataclass
class GenesisDroneAgentConfig(GenesisBaseAgentConfig):
    """Configuration for GenesisDroneAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_drone"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    max_steps: int = 100


@GenesisBaseAgent.register()
class GenesisDroneAgent(GenesisBaseAgent):
    """Agent that evaluates Genesis drone hover policies."""

    agent_type = "genesis_drone"
    config_cls = GenesisDroneAgentConfig
    train_best_metric = "mean_episode_rew_target"

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"follow_target": self.follow_target, "hover": self.hover}

    def follow_target(
        self, target_pos: list[float] | tuple[float, float, float], max_steps: int | None = None
    ) -> dict[str, Any]:
        """Fly the drone to a fixed target position and hover there until the step budget is exhausted.

        Args:
            target_pos: Target [x, y, z] position in world coordinates.
            max_steps: Maximum simulation steps (default: agent max_steps).
        """
        steps = int(max_steps or self._config.max_steps)
        self._robot.set_target_position(target_pos)
        executed_steps = self._run_policy_steps(steps=steps)
        return {"steps": executed_steps, "target_pos": list(target_pos)}

    def hover(self, time_s: float) -> dict[str, Any]:
        """Keep the drone hovering at its current position for the given duration.

        Args:
            time_s: Duration in seconds.
        """
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
