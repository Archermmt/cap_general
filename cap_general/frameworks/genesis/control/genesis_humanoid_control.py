"""Genesis humanoid locomotion control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cap_general.frameworks.genesis.control.genesis_go2_control import GenesisGo2Control, GenesisGo2ControlConfig


@dataclass
class GenesisHumanoidControlConfig(GenesisGo2ControlConfig):
    """Configuration for GenesisHumanoidControl."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "genesis_humanoid"})
    max_steps: int = 500


@GenesisGo2Control.register()
class GenesisHumanoidControl(GenesisGo2Control):
    """Run trained G1 stand, walk, run, and hurdle policies."""

    control_type = "genesis_humanoid"
    config_cls = GenesisHumanoidControlConfig
    train_best_metric = "mean_episode_rew_tracking_lin_vel"

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {"locomote": self.locomote}

    def locomote(self, task: str = "walk", max_steps: int | None = None) -> dict[str, Any]:
        """Run a humanoid locomotion task.

        Args:
            task: One of stand, walk, run, or hurdle.
            max_steps: Maximum policy steps.
        """
        steps = int(max_steps or self._config.max_steps)
        self._robot.set_locomotion_task(task)
        self._run_policy_steps(steps=steps, after_step=lambda: None)
        return {"task": self._robot.task, "steps": steps}

