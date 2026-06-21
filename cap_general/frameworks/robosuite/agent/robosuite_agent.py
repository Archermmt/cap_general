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

@dataclass
class RobosuiteAgentConfig(BaseAgentConfig):
    """Configuration for RobosuiteAgent."""

    robot: dict[str, Any] = field(default_factory=lambda: {"type": "robosuite_robot"})
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    sam3_policy: str = "sam3"
    graspnet_policy: str = "graspnet"
    pyroki_policy: str = "pyroki"
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, -0.107)


@BaseAgent.register()
class RobosuiteAgent(BaseAgent):
    """Agent that executes Franka pick-place code against a Robosuite robot."""

    name = "Robosuite Franka Agent"
    config_cls = RobosuiteAgentConfig

    def __init__(self, config: RobosuiteAgentConfig, logger=None):
        self._sam3_policy = config.sam3_policy
        self._graspnet_policy = config.graspnet_policy
        self._pyroki_policy = config.pyroki_policy
        self._tcp_offset = np.asarray(config.tcp_offset, dtype=np.float64)
        self._ik_cfg = None
        super().__init__(config=config, logger=logger)

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
        }

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get an object's pose from local perception policies.

        Args:
            object_name: Natural-language object name such as ``"red cube"``.
            return_bbox_extent: Whether to return the full oriented-box extent.

        Returns:
            ``(position, quaternion_wxyz, extent_or_none)``.
        """
        import open3d as o3d
        import viser.transforms as vtf

        obs = self._robot._get_robot_obs()
        rgb, depth, intrinsics = self._main_rgbd(obs)
        if self._config.debug:
            self._save_rgbd(rgb, depth)

        depth_2d = depth[:, :, 0] if depth.ndim == 3 else depth
        valid_depth = ~np.isnan(depth_2d)

        results = self._run_policy(self._sam3_policy, method="segment", image=rgb, text_prompt=object_name)
        if not results:
            raise ValueError(f"No SAM3 detections for {object_name!r}")
        if self._config.debug:
            self._save_sam3_results(rgb, object_name, results, function_name="get_object_pose")
        scores = [result.score for result in results]
        best_idx = int(np.argmax(scores))
        mask = np.asarray(results[best_idx].mask, dtype=bool) & valid_depth

        ys_all, xs_all = np.where(valid_depth)
        z_all = depth_2d[ys_all, xs_all]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        points = np.column_stack(((xs_all - cx) * z_all / fx, (ys_all - cy) * z_all / fy, z_all))
        selected = mask[ys_all, xs_all]
        if not np.any(selected):
            raise ValueError(f"Empty SAM3 mask after depth filtering for {object_name!r}")

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points[selected])
        o3d_points, _ = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb = o3d_points.get_oriented_bounding_box()

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs["robot0_robotview"]["pose"][3:]),
            translation=obs["robot0_robotview"]["pose"][:3],
        )
        obb_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3.from_matrix(obb.R),
            translation=obb.center,
        )
        obb_tf_world = cam_extr_tf @ obb_tf

        position = np.asarray(obb_tf_world.wxyz_xyz[-3:], dtype=np.float64)
        quat = np.asarray(obb_tf_world.wxyz_xyz[:4], dtype=np.float64)
        extent = np.asarray(obb.extent, dtype=np.float64)
        return position, quat, extent if return_bbox_extent else None

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object.

        Args:
            object_name: Natural-language object name.

        Returns:
            ``(position, quaternion_wxyz)`` for a gripper grasp pose.
        """
        import viser.transforms as vtf

        obs = self._robot._get_robot_obs()
        rgb, depth, intrinsics = self._main_rgbd(obs)
        if self._config.debug:
            self._save_rgbd(rgb, depth)

        depth_2d = depth[:, :, 0] if depth.ndim == 3 else depth

        results = self._run_policy(self._sam3_policy, method="segment", image=rgb, text_prompt=object_name)
        if not results:
            raise ValueError(f"No SAM3 detections for {object_name!r}")
        if self._config.debug:
            self._save_sam3_results(rgb, object_name, results, function_name="sample_grasp_pose")
        scores = [result.score for result in results]
        best_idx = int(np.argmax(scores))
        segmentation = np.asarray(results[best_idx].mask, dtype=np.int32)
        queried_instance_idx = 1

        grasps = self._run_policy(
            self._graspnet_policy,
            method="plan",
            depth=depth_2d,
            cam_k=intrinsics,
            segmap=segmentation,
            segmap_id=queried_instance_idx,
        )
        grasp_scores = np.asarray(grasps.scores)
        grasp_poses = grasps.grasps

        grasp_sample_tf = vtf.SE3.from_matrix(
            np.asarray(grasp_poses[grasp_scores.argmax()], dtype=np.float64)
        ) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))

        cam_extr_tf = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=obs["robot0_robotview"]["pose"][3:]),
            translation=obs["robot0_robotview"]["pose"][:3],
        )
        grasp_sample_tf_world = cam_extr_tf @ grasp_sample_tf

        return grasp_sample_tf_world.wxyz_xyz[-3:], grasp_sample_tf_world.wxyz_xyz[:4]

    def goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None:
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
        self._robot._set_gripper(1.0)
        for _ in range(30):
            self._robot._step_once()

    def close_gripper(self) -> None:
        """Close gripper fully."""
        self._robot._set_gripper(0.0)
        for _ in range(30):
            self._robot._step_once()

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
        self._robot.move_to_joints_blocking(joints)

    def _compute_reward(self) -> float:
        return float(self._robot.compute_reward())

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
        self._robot.move_to_joints_blocking(joints[:7])

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
    def _pose_from_mask_obb(
        depth_2d: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray, obs: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import open3d as o3d
        from scipy.spatial.transform import Rotation

        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise ValueError("Empty segmentation mask after NaN filtering")
        z = depth_2d[ys, xs]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        points = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb = pcd.get_oriented_bounding_box()

        cam_pose = obs["robot0_robotview"]["pose"]
        cam_rot = Rotation.from_quat([cam_pose[4], cam_pose[5], cam_pose[6], cam_pose[3]])
        center_world = np.asarray(cam_pose[:3], dtype=np.float64) + cam_rot.apply(obb.center)
        rot_world = cam_rot * Rotation.from_matrix(obb.R)
        xyzw = rot_world.as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
        return center_world, wxyz, np.asarray(obb.extent, dtype=np.float64)

    def _save_rgbd(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Save RGBD debug images under the current step directory."""
        from PIL import Image as _Image

        from cap_general.core import utils as cap_utils

        d2d = depth[:, :, 0] if depth.ndim == 3 else depth
        debug_dir = self.debug_dir
        step_cnt = self._robot.step_cnt
        _Image.fromarray(cap_utils.depth_to_rgb(d2d)).save(debug_dir / f"depth_image_{step_cnt}.jpg")
        _Image.fromarray(rgb).save(debug_dir / f"rgb_image_{step_cnt}.jpg")

    def _save_sam3_results(self, rgb: np.ndarray, object_name: str, results: list[Any], *, function_name: str) -> None:
        """Save SAM3 masks and boxes for debugging."""
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from PIL import Image as _Image

        if not results:
            return
        debug_dir = self.debug_dir
        step_cnt = self._robot.step_cnt
        safe_function = self._safe_debug_name(function_name)
        safe_object = self._safe_debug_name(object_name)
        signature = f"{safe_function}_{safe_object}_{step_cnt}"

        image = _Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        image_np = np.array(image)
        top_results = results[:3]

        fig, axes = plt.subplots(1, len(top_results) + 1, figsize=(4 * (len(top_results) + 1), 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1)

        ax_main = axes[0]
        ax_main.imshow(image)
        ax_main.set_title(f"Prompt: '{object_name}'")
        ax_main.axis("off")

        for res in top_results:
            box = self._result_attr(res, "box")
            score = float(self._result_attr(res, "score", 0.0))
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax_main.add_patch(rect)
            ax_main.text(x1, y1, f"{score:.2f}", color="white", fontsize=8, backgroundcolor="red")

        for idx, res in enumerate(top_results, start=1):
            ax = axes[idx]
            mask = np.asarray(self._result_attr(res, "mask"), dtype=bool)
            box = self._result_attr(res, "box")
            score = float(self._result_attr(res, "score", 0.0))
            overlay = image_np.copy()
            color_mask = np.array([30, 144, 255], dtype=np.uint8)
            if mask.shape[:2] == overlay.shape[:2]:
                overlay[mask] = overlay[mask] * 0.5 + color_mask * 0.5
            ax.imshow(overlay)
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="yellow", facecolor="none")
            ax.add_patch(rect)
            ax.set_title(f"Score: {score:.2f}")
            ax.axis("off")
            overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8, copy=False)
            _Image.fromarray(overlay_u8, mode="RGB").save(debug_dir / f"mask_{signature}_{idx}_{score:.2f}.png")

        fig.tight_layout()
        fig.savefig(debug_dir / f"sam3_{signature}.png", format="png")
        plt.close(fig)

    @staticmethod
    def _result_attr(result: Any, name: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    @staticmethod
    def _safe_debug_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_") or "debug"

    def _apply_tcp_offset(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos + Rotation.from_quat(quat_xyzw).apply(self._tcp_offset)

    @staticmethod
    def _matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        xyzw = Rotation.from_matrix(rot).as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
