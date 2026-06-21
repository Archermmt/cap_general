"""Shared Robosuite Franka low-level helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from cap_general.core import utils as cap_utils
from cap_general.core.robot import BaseRobot, BaseRobotConfig

_DEFAULT_CONTROLLER_CFG = str(Path(__file__).resolve().parent / "controllers" / "panda_joint_ctrl.json")


@dataclass
class RobosuiteBaseRobotConfig(BaseRobotConfig):
    """Configuration for RobosuiteBaseRobot."""

    controller_cfg: str = _DEFAULT_CONTROLLER_CFG
    max_steps: int = 1500
    viser_debug: bool = False
    privileged: bool = False
    enable_render: bool = False
    reset_time: float = 0.0


class RobosuiteBaseRobot(BaseRobot):
    """Base class for single-arm Robosuite Franka low-level environments."""

    _SUBSAMPLE_RATE: int = 5
    _ACTION_SLICE: int = -1

    def __init__(
        self,
        config: RobosuiteBaseRobotConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(config=config, logger=logger)
        controller_cfg = Path(config.controller_cfg).expanduser()
        if not controller_cfg.exists():
            controller_cfg = Path(_DEFAULT_CONTROLLER_CFG)
        self.controller_cfg = str(controller_cfg)
        self.max_steps = int(config.max_steps)
        self.viser_debug = bool(config.viser_debug)
        self.privileged = bool(config.privileged)
        self.enable_render = bool(config.enable_render)
        self.save_camera_name = "robot0_robotview"
        self.render_camera_names = [self.save_camera_name]
        self.segmentation_level = "instance"
        self._render_width = 512
        self._render_height = 512
        self._sim_step_count = 0
        self._rng = np.random.default_rng(config.seed)
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._record_wrist_camera = False
        self._wrist_camera_name = "robot0_eye_in_hand"
        self._subsample_rate = self._SUBSAMPLE_RATE
        self._current_joints = np.zeros(7, dtype=np.float64)
        self._gripper_fraction = 1.0

    def _reset(self, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement _reset")

    def _step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs = self._get_robot_obs()
        return obs, 0.0, False, self._step_cnt >= self.max_steps, {}

    def _get_robot_obs(self) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _get_robot_obs")

    @property
    def step_cnt(self) -> int:
        return len(self._frame_buffer)

    def _init_robot_links(self) -> None:
        self.gripper_metric_length = 0.04
        self.base_link_idx = self.robosuite_robot.sim.model.body_name2id("fixed_mount0_base")
        self.gripper_link_idx = self.robosuite_robot.sim.model.body_name2id("gripper0_right_eef")
        self.base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_robot.sim.data.xquat[self.base_link_idx],
                self.robosuite_robot.sim.data.xpos[self.base_link_idx],
            ]
        )
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_robot.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_robot.sim.data.xpos[self.gripper_link_idx],
            ]
        )

    def _init_viser_debug(self, viser_debug: bool) -> None:
        if not viser_debug:
            return
        try:
            import viser
        except ImportError:
            self.viser_debug = False
            return
        self.viser_server = viser.ViserServer()
        self.pyroki_ee_frame_handle = None
        self.mjcf_ee_frame_handle = None
        self.urdf_vis = None
        self.viser_img_handle = None
        self.image_frustum_handle = None
        self.cube_points = None
        self.cube_color = None
        self.cube_center = None
        self.cube_rot = None

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _build_action(self) -> np.ndarray:
        action = np.concatenate([self._current_joints, [self._gripper_fraction, self._gripper_fraction]])
        action[-2:] = 1.0 - action[-2:] * 2.0
        return action

    def _do_robosuite_step(self, action: np.ndarray) -> None:
        sliced = action[: self._ACTION_SLICE] if self._ACTION_SLICE != 0 else action
        need_render = (self._record_frames and self._sim_step_count % self._subsample_rate == 0) or hasattr(
            self, "viser_server"
        )
        if need_render:
            self.robosuite_robot.step(sliced)
        else:
            self.robosuite_robot.step(sliced, skip_render_images=True)

    def _step_once(self) -> None:
        self._do_robosuite_step(self._build_action())
        self._refresh_gripper_pose()
        self._record_frame_if_needed()
        self._sim_step_count += 1

    def move_to_joints_non_blocking(self, joints: np.ndarray) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
        action[-2:] = 1.0 - action[-2:] * 2.0
        self._do_robosuite_step(action)
        self._record_frame_if_needed()
        self._sim_step_count += 1

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float = 0.02,
        max_steps: int = 100,
    ) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target
        for _ in range(max_steps):
            robosuite_obs = self.robosuite_robot._get_observations()
            current = np.asarray(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            if np.linalg.norm(current - target) < tolerance:
                break
            action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
            action[-2:] = 1.0 - action[-2:] * 2.0
            self._do_robosuite_step(action)
            self._record_frame_if_needed()
            self._sim_step_count += 1

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False) -> None:
        self._record_frames = enabled
        self._record_wrist_camera = wrist_camera
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        frame = self.robosuite_robot.sim.render(
            camera_name=self.save_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    def render_wrist(self) -> np.ndarray:
        frame = self.robosuite_robot.sim.render(
            camera_name=self._wrist_camera_name,
            width=self._render_width,
            height=self._render_height,
            depth=False,
        )
        return frame[::-1]

    def _process_camera_observations(
        self, robosuite_obs: dict[str, Any], *, base_wxyz_xyz: np.ndarray | None = None
    ) -> None:
        try:
            import viser.transforms as vtf
            from robosuite.utils.camera_utils import get_real_depth_map
        except ImportError:
            return

        for camera_name in self.render_camera_names:
            camera_entry = robosuite_obs.setdefault(camera_name, {})
            cam_id = self.robosuite_robot.sim.model.camera_name2id(camera_name)
            fovy = self.robosuite_robot.sim.model.cam_fovy[cam_id]
            f = 0.5 * self._render_height / np.tan(fovy * np.pi / 360.0)
            camera_entry["intrinsics"] = np.array(
                [[f, 0, 0.5 * self._render_width], [0, f, 0.5 * self._render_height], [0, 0, 1]]
            )
            base_wxyz_xyz_local = base_wxyz_xyz if base_wxyz_xyz is not None else self.base_link_wxyz_xyz
            cam_world_wxyz_xyz = np.concatenate(
                [
                    vtf.SO3.from_matrix(self.robosuite_robot.sim.data.get_camera_xmat(camera_name)).wxyz,
                    self.robosuite_robot.sim.data.get_camera_xpos(camera_name),
                ]
            )
            cam_robot_tf = (
                (vtf.SE3(wxyz_xyz=base_wxyz_xyz_local).inverse() @ vtf.SE3(wxyz_xyz=cam_world_wxyz_xyz))
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, np.pi, 0.0),
                    translation=np.array([0, 0, 0]),
                )
                @ vtf.SE3.from_rotation_and_translation(
                    rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi),
                    translation=np.array([0, 0, 0]),
                )
            )
            camera_entry["pose"] = np.concatenate([cam_robot_tf.translation(), cam_robot_tf.rotation().wxyz])
            camera_entry["pose_mat"] = cam_robot_tf.as_matrix()
            camera_entry["images"] = {}
            if f"{camera_name}_image" in robosuite_obs:
                camera_entry["images"]["rgb"] = robosuite_obs[f"{camera_name}_image"][::-1]
            if f"{camera_name}_depth" in robosuite_obs:
                camera_entry["images"]["depth"] = get_real_depth_map(
                    self.robosuite_robot.sim,
                    robosuite_obs[f"{camera_name}_depth"][::-1],
                )

    def _compute_gripper_obs(self, robosuite_obs: dict[str, Any]) -> None:
        gripper_metric_length = getattr(self, "gripper_metric_length", 0.04)
        gripper = robosuite_obs.get("robot0_gripper_qpos", [self._gripper_fraction])
        robosuite_obs["robot_joint_pos"] = np.concatenate(
            [robosuite_obs["robot0_joint_pos"], [gripper[0] / gripper_metric_length]]
        )
        eef_pos = robosuite_obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = robosuite_obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))
        robosuite_obs["robot_cartesian_pos"] = np.concatenate([eef_pos, eef_quat, [gripper[0] / gripper_metric_length]])

    def _record_frame_if_needed(self) -> None:
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()

    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        frame = self.render()
        self._frame_buffer.append(frame)
        if self._record_wrist_camera:
            self._wrist_frame_buffer.append(self.render_wrist())

    def _refresh_gripper_pose(self) -> None:
        if hasattr(self, "gripper_link_idx"):
            self.gripper_link_wxyz_xyz = np.concatenate(
                [
                    self.robosuite_robot.sim.data.xquat[self.gripper_link_idx],
                    self.robosuite_robot.sim.data.xpos[self.gripper_link_idx],
                ]
            )

    def compute_reward(self) -> float:
        return 0.0
