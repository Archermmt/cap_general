"""Genesis Franka agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cap_general.core.agent import BaseAgent, BaseAgentConfig

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class FrankaAgentConfig(BaseAgentConfig):
    """Configuration for FrankaAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_franka"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    horizon: int = 100


@BaseAgent.register()
class FrankaAgent(BaseAgent):
    """Agent that runs high-level Franka episodes through the Genesis robot controller."""

    name = "Genesis Franka Agent"
    config_cls = FrankaAgentConfig

    def __init__(self, config: FrankaAgentConfig, logger: Logger):
        self.horizon = int(config.horizon)
        super().__init__(config=config, logger=logger)

    @classmethod
    def agent_type(cls) -> str:
        return "franka"

    def init_genesis(self, gs_scene: Any) -> None:
        self._robot.init_genesis(gs_scene)

    def _execute_rules(self) -> str:
        """Return valid rules for execute for the Genesis Franka robot."""
        return (
            "The Genesis Franka scene is controlled by robot. It contains one Franka arm "
            "and configured objects such as boxes, spheres, or cylinders. Generated code should "
            "use robot methods such as robot.step(...), robot.get_observation(...), "
            "robot.set_joint_positions(...), robot.move_to_pose(...), robot.grasp(), "
            "robot.release(), and robot.step_simulation(). Do not create scenes, robots, "
            "or objects in execute code."
        )

    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return Genesis Franka functions exposed to generated code."""
        return {"franka_episode": self.franka_episode}

    def franka_episode(
        self,
        actions: list[dict[str, Any]] | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run a Franka episode using robot-controlled scene logic.

        Args:
            actions: Optional low-level robot action dictionaries. When omitted,
                the episode advances the simulator with empty actions.
            max_steps: Maximum number of robot control steps.

        Returns:
            Episode summary with step count and final observation.
        """
        steps = int(max_steps or self.horizon)
        obs = self._robot.last_obs
        action_sequence = actions or []

        for step_idx in range(steps):
            action = action_sequence[step_idx] if step_idx < len(action_sequence) else {}
            obs, _reward, _terminated, _truncated, _info = self._robot.step(action)

        return {
            "steps": step_idx + 1 if steps > 0 else 0,
            "obs": obs,
        }
