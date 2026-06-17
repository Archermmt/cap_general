"""CAP Robosuite environment wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat

from cap_general.core.env import BaseEnv, BaseEnvConfig
from cap_general.frameworks.robosuite.env.robosuite_cube_env import (
    MockRobosuiteCubeEnv,
    RobosuiteCubeEnv,
)


@dataclass
class RobosuiteEnvConfig(BaseEnvConfig):
    """Configuration for RobosuiteEnv."""

    low_level: str = "robosuite_cube_env"
    privileged: bool = False
    enable_render: bool = True
    viser_debug: bool = False
    max_steps: int = 1500
    controller_cfg: str = "cap_general/frameworks/robosuite/controllers/panda_joint_ctrl.json"
    mock_fallback: bool = True


@BaseEnv.register()
class RobosuiteEnv(BaseEnv):
    """Gymnasium-style wrapper around local Robosuite low-level envs."""

    name = "Robosuite Env"
    config_cls = RobosuiteEnvConfig

    def __init__(self, config: RobosuiteEnvConfig, logger: logging.Logger | None = None):
        super().__init__(config=config, logger=logger)
        self._config = config
        self.low_level_env = self._build_low_level()

    @classmethod
    def env_type(cls) -> str:
        return "robosuite"

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.low_level_env.reset(seed=self._seed, options=options)

    def _step(self, action: Any) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        if isinstance(action, str):
            raise TypeError("RobosuiteEnv expects low-level actions; code execution belongs to RobosuiteAgent")
        obs, reward, terminated, truncated, info = self.low_level_env.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def get_observation(self, folder: str | Path) -> dict[str, Any]:
        return self.low_level_env.get_observation()

    def record(self, folder: str | Path, start_frm: int = 0, end_frm: int | None = None) -> dict:
        if not hasattr(self.low_level_env, "get_video_frames_range"):
            return {"videos": {}, "main_video": None}
        frames = self.low_level_env.get_video_frames_range(start_frm, end_frm or self.step_cnt)
        if not frames:
            return {"videos": {}, "main_video": None}
        from cap_general.core import utils as cap_utils

        record_path = Path(folder)
        record_path.mkdir(parents=True, exist_ok=True)
        main_video = cap_utils.save_video(record_path / "robot0_robotview.mp4", frames)
        return {"videos": {"robot0_robotview": main_video}, "main_video": main_video}

    def compute_reward(self) -> float:
        if hasattr(self.low_level_env, "compute_reward"):
            return float(self.low_level_env.compute_reward())
        return 0.0

    def task_completed(self) -> bool | None:
        if hasattr(self.low_level_env, "task_completed"):
            return bool(self.low_level_env.task_completed())
        return None

    def _build_low_level(self):
        try:
            if self._config.low_level not in {
                "robosuite_cube_env",
                "robosuite_cude_env",
                "franka_robosuite_cubes_low_level",
            }:
                raise ValueError(f"Unknown local Robosuite low-level env: {self._config.low_level}")
            return RobosuiteCubeEnv(
                controller_cfg=self._config.controller_cfg,
                max_steps=self._config.max_steps,
                seed=self._seed,
                viser_debug=self._config.viser_debug,
                privileged=self._config.privileged,
                enable_render=self._config.enable_render,
            )
        except Exception as exc:
            if not self._config.mock_fallback:
                raise
            self.logger.warning("Falling back to mock Robosuite cube env: %s", exc)
            return MockRobosuiteCubeEnv()
