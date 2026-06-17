"""Robosuite code-execution agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from cap_general.core.agent import BaseAgent, BaseAgentConfig

PROMPT = """
You are controlling a Franka Emika robot with the API described below.
Goal: Pick up the red cube and gently stack it on top of the green cube, then release it.

Key rules:
- The extent from get_object_pose(..., return_bbox_extent=True) is the FULL side length. Use extent[2]/2 for half-height.
- For placement orientation, reuse the grasp quaternion from sample_grasp_pose. Do NOT use the quaternion from get_object_pose (it is unreliable for orientation).
- Always use z_approach=0.1 when approaching an object for grasping or placing.
- After grasping, lift the cube to a safe height (at least +0.2m in Z) before moving laterally to the placement location.
- The stacking height formula is: place_z = green_center_z + green_extent[2]/2 + red_extent[2]/2
- Nothing should be dropped from a height. Always approach with z_approach for controlled descent.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""

ORACLE_CODE = """
import numpy as np

_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
_, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)

# Sample a grasp pose for the red cube and pick it up
pick_pos, pick_quat = sample_grasp_pose("red cube")
goto_pose(pick_pos, pick_quat, z_approach=0.1)
close_gripper()
# Lift the red cube after grasping
post_pick_pos = pick_pos.copy()
post_pick_pos[2] += 0.2
goto_pose(post_pick_pos, pick_quat)

# Compute placement pose on top of the green cube
green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)

place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2]/2 + red_ext[2]/2
# Use down orientation for placement
place_quat = np.array([0.0, 0.0, 1.0, 0.0])

# Approach and place the red cube on the green cube
goto_pose(place_pos, pick_quat, z_approach=0.1)
open_gripper()

# Retract after placing
post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, place_quat)
RESULT = {"success": True}
"""


@dataclass
class RobosuiteAgentConfig(BaseAgentConfig):
    """Configuration for RobosuiteAgent."""

    env: dict[str, Any] = field(default_factory=lambda: {"type": "robosuite"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    sam3_policy: str = "sam3"
    graspnet_policy: str = "graspnet"
    pyroki_policy: str = "pyroki"
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, -0.107)


@BaseAgent.register()
class RobosuiteAgent(BaseAgent):
    """Agent that executes Franka pick-place code against a Robosuite env."""

    name = "Robosuite Franka Agent"
    config_cls = RobosuiteAgentConfig
    oracle_code = ORACLE_CODE

    def __init__(self, config: RobosuiteAgentConfig):
        self._sam3_policy = config.sam3_policy
        self._graspnet_policy = config.graspnet_policy
        self._pyroki_policy = config.pyroki_policy
        self._tcp_offset = np.asarray(config.tcp_offset, dtype=np.float64)
        self._ik_cfg = None
        super().__init__(config=config)

    @classmethod
    def agent_type(cls) -> str:
        return "robosuite"

    def _execute_rules(self) -> str:
        """Return Cap-X Franka pick-place prompt and API docs."""
        return f"{PROMPT.strip()}\n\nAPIs:\n{self._function_doc()}"

    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return FrankaControlApi-compatible helpers."""
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "home_pose": self.home_pose,
            "oracle_code": self.get_oracle_code,
        }

    def get_oracle_code(self) -> str:
        """Return the Cap-X FrankaPickPlaceCodeEnv oracle code."""
        return self.oracle_code

    def get_object_pose(
        self,
        object_name: str,
        return_bbox_extent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get an object's pose from privileged state or local perception policies.

        Args:
            object_name: Natural-language object name such as ``"red cube"``.
            return_bbox_extent: Whether to return the full oriented-box extent.

        Returns:
            ``(position, quaternion_wxyz, extent_or_none)``.
        """
        privileged = self._privileged_object_pose(object_name)
        if privileged is not None:
            position, quat, extent = privileged
            return position, quat, extent if return_bbox_extent else None
        # Perception fallback: keep the interface wired to local policies, while
        # letting policy errors surface with useful context when models are configured.
        obs = self._env.get_observation(self._record_dir / self.step_dir)
        rgb, depth, intrinsics = self._main_rgbd(obs)
        masks = self._run_policy(self._sam3_policy, method="segment", image=rgb, text_prompt=object_name)
        if not masks:
            raise ValueError(f"No SAM3 detections for {object_name!r}")
        mask = np.asarray(masks[0].mask, dtype=bool)
        position, quat, extent = self._pose_from_mask(depth, intrinsics, mask)
        return position, quat, extent if return_bbox_extent else None

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object.

        Args:
            object_name: Natural-language object name.

        Returns:
            ``(position, quaternion_wxyz)`` for a gripper grasp pose.
        """
        privileged = self._privileged_object_pose(object_name)
        if privileged is not None:
            position, quat, _extent = privileged
            return position.copy(), quat.copy()
        obs = self._env.get_observation(self._record_dir / self.step_dir)
        rgb, depth, intrinsics = self._main_rgbd(obs)
        masks = self._run_policy(self._sam3_policy, method="segment", image=rgb, text_prompt=object_name)
        if not masks:
            raise ValueError(f"No SAM3 detections for {object_name!r}")
        mask = np.asarray(masks[0].mask, dtype=np.int32)
        grasps = self._run_policy(
            self._graspnet_policy,
            method="plan",
            depth=depth[:, :, 0] if depth.ndim == 3 else depth,
            cam_k=intrinsics,
            segmap=mask,
            segmap_id=1,
        )
        best_idx = int(np.asarray(grasps.scores).argmax())
        pose = np.asarray(grasps.grasps[best_idx], dtype=np.float64)
        return pose[:3, 3], self._matrix_to_wxyz(pose[:3, :3])

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Move to a Cartesian pose using the configured PyRoKi policy.

        Args:
            position: Target XYZ in meters.
            quaternion_wxyz: Target orientation as WXYZ quaternion.
            z_approach: Optional approach offset before moving to the final pose.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = self._apply_tcp_offset(pos, quat)
        if z_approach:
            self._goto_offset_pose(offset_pos + np.array([0.0, 0.0, -z_approach]), quat)
        self._goto_offset_pose(offset_pos, quat)

    def open_gripper(self) -> None:
        """Open gripper fully."""
        self._env.low_level_env._set_gripper(1.0)
        for _ in range(30):
            self._env.low_level_env._step_once()

    def close_gripper(self) -> None:
        """Close gripper fully."""
        self._env.low_level_env._set_gripper(0.0)
        for _ in range(30):
            self._env.low_level_env._step_once()

    def home_pose(self) -> None:
        """Move the robot to a safe home pose."""
        joints = np.array(
            [
                -2.95353726e-02,
                1.69197371e-01,
                2.39244731e-03,
                -2.64089311e00,
                -2.01237851e-03,
                2.94565778e00,
                8.31390616e-01,
            ],
            dtype=np.float64,
        )
        self._env.low_level_env.move_to_joints_blocking(joints)

    def _compute_reward(self) -> float:
        return float(self._env.compute_reward())

    def _goto_offset_pose(self, offset_pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        if self._pyroki_policy not in self._policies:
            joints = np.zeros(7, dtype=np.float64)
        else:
            result = self._run_policy(
                self._pyroki_policy,
                method="solve_ik",
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, offset_pos]),
                prev_cfg=self._ik_cfg,
            )
            joints = np.asarray(result.joint_positions, dtype=np.float64)
        self._ik_cfg = joints
        self._env.low_level_env.move_to_joints_blocking(joints[:7])

    def _privileged_object_pose(
        self,
        object_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        low_level = self._env.low_level_env
        if hasattr(low_level, "object_pose"):
            return tuple(np.asarray(v, dtype=np.float64) for v in low_level.object_pose(object_name))
        obs = low_level.get_observation()
        objects = obs.get("objects")
        if isinstance(objects, dict) and object_name in objects:
            obj = objects[object_name]
            return (
                np.asarray(obj["position"], dtype=np.float64),
                np.asarray(obj.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64),
                np.asarray(obj.get("extent", [0.04, 0.04, 0.04]), dtype=np.float64),
            )
        return None

    @staticmethod
    def _main_rgbd(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera = obs.get("robot0_robotview", {})
        images = camera.get("images", {})
        rgb = np.asarray(images.get("rgb"))
        depth = np.asarray(images.get("depth"))
        intrinsics = np.asarray(camera.get("intrinsics"))
        if rgb.size == 0 or depth.size == 0 or intrinsics.size == 0:
            raise ValueError("Robosuite observation does not contain robot0_robotview rgb/depth/intrinsics")
        return rgb, depth, intrinsics

    @staticmethod
    def _pose_from_mask(
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise ValueError("Empty segmentation mask")
        z_img = depth[:, :, 0] if depth.ndim == 3 else depth
        z = z_img[ys, xs]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        points = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
        center = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        return center, np.array([1.0, 0.0, 0.0, 0.0]), extent

    def _apply_tcp_offset(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos + Rotation.from_quat(quat_xyzw).apply(self._tcp_offset)

    @staticmethod
    def _matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        xyzw = Rotation.from_matrix(rot).as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
