"""Robosuite Franka cube-stack low-level environment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from cap_general.core.robot import BaseRobot
from cap_general.frameworks.robosuite.robot.robosuite_base_robot import RobosuiteBaseRobot, RobosuiteBaseRobotConfig


@dataclass
class RobosuiteCubeRobotConfig(RobosuiteBaseRobotConfig):
    """Configuration for RobosuiteCubeRobot."""


@BaseRobot.register()
class RobosuiteCubeRobot(RobosuiteBaseRobot):
    """Robosuite Franka Stack environment compatible with FrankaControlApi."""

    name = "Robosuite Cube Env"
    config_cls = RobosuiteCubeRobotConfig
    _SUBSAMPLE_RATE = 5

    @classmethod
    def robot_type(cls) -> str:
        return "robosuite_robot"

    def __init__(
        self,
        config: RobosuiteCubeRobotConfig | None = None,
        logger: logging.Logger,
    ) -> None:
        if config is None:
            config = RobosuiteCubeRobotConfig()
        super().__init__(config=config, logger=logger)
        try:
            self._init_robosuite_stack()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize RobosuiteCubeRobot: {exc}") from exc

    def _init_robosuite_stack(self) -> None:
        import robosuite as suite
        from robosuite.controllers.composite.composite_controller_factory import (
            load_composite_controller_config,
        )
        from robosuite.utils.placement_samplers import UniformRandomSampler

        controller = load_composite_controller_config(controller=self.controller_cfg)
        if self.privileged and not self.enable_render:
            self.render_camera_names = []
            self.robosuite_robot = suite.environments.manipulation.stack.Stack(
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
            self.robosuite_robot = suite.environments.manipulation.stack.Stack(
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

        cubes = [self.robosuite_robot.cubeA, self.robosuite_robot.cubeB]
        self.robosuite_robot.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cubes,
            x_range=[-0.18, 0.18],
            y_range=[-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.robosuite_robot.table_offset,
            z_offset=0.01,
            rng=self.robosuite_robot.rng,
        )
        self._init_robot_links()
        self._init_viser_debug(self.viser_debug)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        return BaseRobot.reset(self, options=options)

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._seed is not None:
            self.robosuite_robot.rng = np.random.default_rng(self._seed)
            self.robosuite_robot.placement_initializer.rng = self.robosuite_robot.rng
        self.robosuite_robot.reset()
        self.robosuite_robot.sim.data.qpos[6] -= np.pi
        self._sim_step_count = 0
        for _ in range(50):
            self.robosuite_robot.sim.forward()
            self.robosuite_robot.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_robot._get_observations()
        self._current_joints = np.asarray(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi
        self._refresh_gripper_pose()
        if self._video_fmt:
            self.enable_video_capture(True, clear=True)
        obs = self._get_robot_obs()
        return obs, {
            "task_prompt": "Place the primary cube on top of the secondary cube. Quaternions are WXYZ.",
            "options": options or {},
        }

    def object_pose(self, name: str):
        obs = self._get_robot_obs()
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
            pose_dict[key] = [float(x) for x in np.concatenate([robosuite_obs[pos_key], robosuite_obs[quat_key]])]
        return pose_dict

    def compute_reward(self) -> float:
        return float(self.robosuite_robot.reward(action=None))

    def _get_robot_obs(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_robot._get_observations()
        pose_dict = self._cube_pose_dict(robosuite_obs)
        robosuite_obs["cube_poses"] = {key: np.asarray(value, dtype=np.float32) for key, value in pose_dict.items()}
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
