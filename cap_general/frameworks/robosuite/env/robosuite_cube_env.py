"""Robosuite Franka cube-stack low-level environment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.env import BaseEnv
from cap_general.frameworks.robosuite.env.robosuite_base_env import RobosuiteBaseEnv, RobosuiteBaseEnvConfig


@dataclass
class RobosuiteCubeEnvConfig(RobosuiteBaseEnvConfig):
    """Configuration for RobosuiteCubeEnv.

    ``low_level`` and ``mock_fallback`` are accepted for compatibility with
    older wrapper-style yaml files.
    """

    low_level: str = "robosuite_cube_env"
    mock_fallback: bool = False


class MockRobosuiteCubeEnv:
    """Mock fallback matching the subset used by Franka pick-place oracle code."""

    max_steps = 1500

    def __init__(self, *args, **kwargs):
        self._sim_step_count = 0
        self._step_count = 0
        self._gripper_fraction = 1.0
        self._current_joints = np.zeros(7, dtype=np.float64)
        self._objects = {
            "green cube": {
                "position": np.array([0.48, -0.08, 0.82], dtype=np.float64),
                "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "extent": np.array([0.04, 0.04, 0.04], dtype=np.float64),
            },
            "red cube": {
                "position": np.array([0.48, 0.08, 0.82], dtype=np.float64),
                "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                "extent": np.array([0.04, 0.04, 0.04], dtype=np.float64),
            },
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._sim_step_count = 0
        self._step_count = 0
        self._gripper_fraction = 1.0
        return self.get_observation(), {"mock": True, "seed": seed, "options": options or {}}

    def get_observation(self) -> dict[str, Any]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        depth = np.ones((64, 64, 1), dtype=np.float32)
        intrinsics = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]])
        return {
            "robot0_robotview": {
                "pose": np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                "pose_mat": np.eye(4),
                "intrinsics": intrinsics,
                "images": {"rgb": image, "depth": depth},
            },
            "robot_joint_pos": np.concatenate([self._current_joints, [self._gripper_fraction]]),
            "robot_cartesian_pos": np.array(
                [0.5, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0, self._gripper_fraction]
            ),
            "objects": self._objects,
        }

    def object_pose(self, name: str):
        obj = self._objects[name]
        return obj["position"].copy(), obj["quat"].copy(), obj["extent"].copy()

    def move_to_joints_blocking(self, joints, *, tolerance: float = 0.02, max_steps: int = 100):
        self._current_joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._sim_step_count += 1

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        self._sim_step_count += 1

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        return self.get_observation(), self.compute_reward(), False, False, {}

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def render(self, mode: str = "rgb_array"):
        return self.get_observation()["robot0_robotview"]["images"]["rgb"]

    def get_video_frames_range(self, start: int, end: int):
        return []


@BaseEnv.register()
class RobosuiteCubeEnv(RobosuiteBaseEnv):
    """Robosuite Franka Stack environment compatible with FrankaControlApi."""

    name = "Robosuite Cube Env"
    config_cls = RobosuiteCubeEnvConfig
    _SUBSAMPLE_RATE = 5

    @classmethod
    def env_type(cls) -> str:
        return "robosuite"

    def __init__(
        self,
        config: RobosuiteCubeEnvConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if config is None:
            config = RobosuiteCubeEnvConfig()
        super().__init__(config=config, logger=logger)
        self._mock_env: MockRobosuiteCubeEnv | None = None
        try:
            self._init_robosuite_stack()
        except Exception as exc:
            if config.mock_fallback:
                self.logger.warning("Falling back to mock Robosuite cube env: %s", exc)
                self._mock_env = MockRobosuiteCubeEnv()
                return
            raise RuntimeError(f"Failed to initialize RobosuiteCubeEnv: {exc}") from exc

    def _init_robosuite_stack(self) -> None:
        import robosuite as suite
        from robosuite.controllers.composite.composite_controller_factory import (
            load_composite_controller_config,
        )
        from robosuite.utils.placement_samplers import UniformRandomSampler

        controller = load_composite_controller_config(controller=self.controller_cfg)
        if self.privileged and not self.enable_render:
            self.render_camera_names = []
            self.robosuite_env = suite.environments.manipulation.stack.Stack(
                robots=["Panda"],
                use_camera_obs=False,
                has_renderer=False,
                has_offscreen_renderer=False,
                camera_names=self.render_camera_names,
                renderer="mujoco",
                reward_shaping=True,
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=controller,
                horizon=self.max_steps,
            )
        else:
            self.robosuite_env = suite.environments.manipulation.stack.Stack(
                robots=["Panda"],
                has_renderer=not self.privileged,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                camera_depths=True,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=controller,
                horizon=self.max_steps,
                reward_shaping=True,
            )

        cubes = [self.robosuite_env.cubeA, self.robosuite_env.cubeB]
        self.robosuite_env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cubes,
            x_range=[-0.18, 0.18],
            y_range=[-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.robosuite_env.table_offset,
            z_offset=0.01,
            rng=self.robosuite_env.rng,
        )
        self._init_robot_links()
        self._init_viser_debug(self.viser_debug)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        return BaseEnv.reset(self, options=options)

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._mock_env is not None:
            return self._mock_env.reset(seed=self._seed, options=options)
        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi
        self._step_count = 0
        self._sim_step_count = 0
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.asarray(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi
        self._refresh_gripper_pose()
        if self._video_fmt:
            self.enable_video_capture(True, clear=True)
        obs = self.get_observation()
        return obs, {
            "task_prompt": "Place the primary cube on top of the secondary cube. Quaternions are WXYZ.",
            "options": options or {},
        }

    def object_pose(self, name: str):
        if self._mock_env is not None:
            return self._mock_env.object_pose(name)
        obs = self.get_observation()
        aliases = {
            "red cube": "primary",
            "green cube": "secondary",
            "primary cube": "primary",
            "secondary cube": "secondary",
        }
        key = aliases.get(name, name)
        if key not in obs["cube_poses"]:
            raise KeyError(f"Unknown cube object {name!r}")
        pose = np.asarray(obs["cube_poses"][key], dtype=np.float64)
        extent = np.asarray(obs.get("cube_extents", {}).get(key, [0.04, 0.04, 0.04]), dtype=np.float64)
        return pose[:3], pose[3:7], extent

    def _cube_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        pose_dict = {}
        for key, pos_key, quat_key in (
            ("primary", "cubeA_pos", "cubeA_quat"),
            ("secondary", "cubeB_pos", "cubeB_quat"),
        ):
            pose_dict[key] = [
                float(x)
                for x in np.concatenate(
                    [robosuite_obs[pos_key], robosuite_obs[quat_key]]
                )
            ]
        return pose_dict

    def compute_reward(self) -> float:
        if self._mock_env is not None:
            return float(self._mock_env.compute_reward())
        return float(self.robosuite_env.reward(action=None))

    def task_completed(self) -> bool:
        if self._mock_env is not None:
            return bool(self._mock_env.task_completed())
        return bool(self.robosuite_env._check_success())

    def get_observation(self, folder: str | Path | None = None) -> dict[str, Any]:
        if self._mock_env is not None:
            return self._mock_env.get_observation()
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._cube_pose_dict(robosuite_obs)
        robosuite_obs["cube_poses"] = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in pose_dict.items()
        }
        robosuite_obs["cube_extents"] = {
            "primary": np.array([0.04, 0.04, 0.04], dtype=np.float32),
            "secondary": np.array([0.04, 0.04, 0.04], dtype=np.float32),
        }
        robosuite_obs["objects"] = {
            "red cube": {
                "position": robosuite_obs["cube_poses"]["primary"][:3],
                "quat": robosuite_obs["cube_poses"]["primary"][3:7],
                "extent": robosuite_obs["cube_extents"]["primary"],
            },
            "green cube": {
                "position": robosuite_obs["cube_poses"]["secondary"][:3],
                "quat": robosuite_obs["cube_poses"]["secondary"][3:7],
                "extent": robosuite_obs["cube_extents"]["secondary"],
            },
        }
        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs

    def _set_gripper(self, fraction: float) -> None:
        if self._mock_env is not None:
            self._mock_env._set_gripper(fraction)
            return
        super()._set_gripper(fraction)

    def _step_once(self) -> None:
        if self._mock_env is not None:
            self._mock_env._step_once()
            return
        super()._step_once()

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.02,
        max_steps: int = 100,
    ) -> None:
        if self._mock_env is not None:
            self._mock_env.move_to_joints_blocking(joints, tolerance=tolerance, max_steps=max_steps)
            return
        super().move_to_joints_blocking(joints, tolerance=tolerance, max_steps=max_steps)

    def get_video_frames_range(self, start: int, end: int):
        if self._mock_env is not None:
            return self._mock_env.get_video_frames_range(start, end)
        return super().get_video_frames_range(start, end)


# Typo-compatible alias for callers that used "cude" from the request text.
RobosuiteCudeEnv = RobosuiteCubeEnv

__all__ = ["MockRobosuiteCubeEnv", "RobosuiteCubeEnv", "RobosuiteCudeEnv", "RobosuiteCubeEnvConfig"]
