"""Genesis GO2 locomotion agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cap_general.core.agent import BaseAgent, BaseAgentConfig

if TYPE_CHECKING:
    from logging import Logger


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

    name = "Genesis GO2 Agent"
    config_cls = Go2AgentConfig

    def __init__(self, config: Go2AgentConfig, logger: Logger):
        self._policy_name = config.policy
        self.horizon = int(config.horizon)
        super().__init__(config=config, logger=logger)

    @classmethod
    def agent_type(cls) -> str:
        return "genesis_go2"

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
        steps = int(max_steps or self.horizon)
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
            action = self._run_policy(self._policy_name, obs=obs)
            action = self._robot.apply_turn_to_action(action, turn_angle)
            obs, _reward, terminated, truncated, _info = self._robot.step(action)
            after_step()
            if terminated or truncated:
                break
            obs = self._robot.policy_obs
        return None
