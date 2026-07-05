"""Genesis GO2 locomotion agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cap_general.frameworks.genesis.agent.genesis_base_agent import GenesisBaseAgent, GenesisBaseAgentConfig


@dataclass
class GenesisGo2AgentConfig(GenesisBaseAgentConfig):
    """Configuration for GenesisGo2Agent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_go2"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy: str = "runner"
    max_steps: int = 100


@GenesisBaseAgent.register()
class GenesisGo2Agent(GenesisBaseAgent):
    """Agent that evaluates Genesis GO2 locomotion policies."""

    agent_type = "genesis_go2"
    config_cls = GenesisGo2AgentConfig
    train_best_metric = "mean_episode_rew_tracking_lin_vel"

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"walk_forward": self.walk_forward, "stand_still": self.stand_still}

    def walk_forward(self, max_steps: int | None = None, turn_angle: float = 0.0) -> dict[str, Any]:
        """Make GO2 walk forward and optionally turn by a yaw angle.

        Args:
            max_steps: Maximum simulation steps (default: agent max_steps).
            turn_angle: Yaw bias in radians applied to policy actions during the walk.
        """
        steps = int(max_steps or self._config.max_steps)
        self._robot.set_walk_command(turn_angle=0.0, steps=steps)
        self._run_policy_steps(
            steps=steps,
            after_step=lambda: self._robot.set_walk_command(turn_angle=0.0, steps=steps),
            turn_angle=float(turn_angle),
        )
        return {"steps": steps, "turn_angle": float(turn_angle)}

    def stand_still(self, time_s: float) -> dict[str, Any]:
        """Keep GO2 standing still for the given duration.

        Args:
            time_s: Duration in seconds.
        """
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
