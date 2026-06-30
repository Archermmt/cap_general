"""Local Contact-GraspNet model implementation."""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cap_general.core.policy.base_policy import BasePolicy

from .base_policy import BasePolicyConfig


@dataclass
class GraspNetResult:
    """Contact-GraspNet grasp prediction result."""

    grasps: np.ndarray
    scores: np.ndarray
    contact_points: np.ndarray


def _recursive_key_value_assign(data: dict, keys: list[Any], value: Any):
    if len(keys) > 1:
        _recursive_key_value_assign(data[keys[0]], keys[1:], value)
    elif keys:
        data[keys[0]] = value


def load_contact_graspnet_config(
    checkpoint_root: str | Path,
    batch_size: int | None = None,
    max_epoch: int | None = None,
    data_path: str | None = None,
    arg_configs: list[str] | None = None,
) -> dict[str, Any]:
    """Load Contact-GraspNet YAML config locally."""

    config_path = Path(checkpoint_root) / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    for conf in arg_configs or []:
        key_str, value = conf.split(":", 1)
        with_context_value = value
        try:
            with_context_value = eval(value)  # noqa: S307 - mirrors upstream config parsing
        except Exception:
            pass
        keys = [int(k) if k.isdigit() else k for k in key_str.split(".")]
        _recursive_key_value_assign(config, keys, with_context_value)

    if batch_size is not None:
        config["OPTIMIZER"]["batch_size"] = int(batch_size)
    if max_epoch is not None:
        config["OPTIMIZER"]["max_epoch"] = int(max_epoch)
    if data_path is not None:
        config["DATA"]["data_path"] = data_path
    config["DATA"]["classes"] = None
    return config


def _sample_random_camera_viewpoint(
    target_point: np.ndarray, xy_extent_meters: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random camera viewpoint looking at target_point. Returns (position, wxyz)."""
    import viser.transforms as vtf

    position = np.random.uniform(-xy_extent_meters, xy_extent_meters, 3)
    z_c = target_point - position
    z_c /= np.linalg.norm(z_c)
    down_world = np.array([0.0, 1.0, 0.0])
    y_c = down_world - np.dot(down_world, z_c) * z_c
    y_c_norm = np.linalg.norm(y_c)
    if y_c_norm < 1e-6:
        y_c = np.array([1.0, 0.0, 0.0])
        y_c = y_c - np.dot(y_c, z_c) * z_c
        y_c /= np.linalg.norm(y_c)
    else:
        y_c /= y_c_norm
    x_c = np.cross(y_c, z_c)
    R_wc = np.column_stack([x_c, y_c, z_c])
    return position, vtf.SO3.from_matrix(R_wc).wxyz


@dataclass
class GraspNetPolicyConfig(BasePolicyConfig):
    """Configuration for GraspNetPolicy."""

    vendor_root: str | Path
    checkpoint_root: str | Path | None = None
    checkpoint_dir: str | Path | None = None
    checkpoint_name: str = "model.pt"
    device: str = "cuda"
    describe: str = field(
        default=(
            "Predicts 6-DoF object grasps from RGB-D-derived point clouds, depth, "
            "camera intrinsics, and segmentation masks using Contact-GraspNet."
        ),
        kw_only=True,
    )


@BasePolicy.register()
class GraspNetPolicy(BasePolicy):
    """Local Contact-GraspNet grasp planning model."""

    name = "GraspNet Policy"
    config_cls = GraspNetPolicyConfig

    def __init__(self, config: GraspNetPolicyConfig, logger: logging.Logger):
        super().__init__(config=config, logger=logger)
        self._vendor_root = Path(config.vendor_root)
        self._checkpoint_root = (
            Path(config.checkpoint_root)
            if config.checkpoint_root is not None
            else self._vendor_root / "checkpoints" / "contact_graspnet"
        )
        self._checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir is not None else self._checkpoint_root / "checkpoints"
        )
        self._checkpoint_name = config.checkpoint_name
        self._device = config.device
        self._estimator = None

    @classmethod
    def policy_type(cls) -> str:
        return "graspnet"

    def reset(self) -> None:
        """Load Contact-GraspNet locally if needed."""
        if self._estimator is not None:
            return

        pointnet_root = self._vendor_root / "Pointnet_Pointnet2_pytorch"
        for path in (pointnet_root, self._vendor_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)

        try:
            from contact_graspnet_pytorch.checkpoints import CheckpointIO
            from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
        except ImportError as exc:
            raise ImportError(
                f"GraspNetPolicy requires contact_graspnet_pytorch. Checked vendor_root={self._vendor_root}"
            ) from exc

        config = load_contact_graspnet_config(self._checkpoint_root)
        self._estimator = GraspEstimator(config)
        checkpoint_io = CheckpointIO(
            checkpoint_dir=str(self._checkpoint_dir),
            model=self._estimator.model,
        )
        try:
            checkpoint_io.load(self._checkpoint_name)
        except FileExistsError:
            pass
        self._estimator.model.float()
        self._estimator.model.eval()

        import torch

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def plan(
        self,
        depth: np.ndarray,
        cam_k: np.ndarray,
        segmap: np.ndarray,
        segmap_id: int,
        local_regions: bool = True,
        filter_grasps: bool = True,
        skip_border_objects: bool = False,
        z_range: list[float] | None = None,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> GraspNetResult:
        """Plan grasps from depth, camera intrinsics, and segmentation map."""
        import torch
        import viser.transforms as vtf

        self.reset()
        z_range = [0.2, 2.0] if z_range is None else z_range

        estimator_device = self._estimator.device
        device_type = getattr(estimator_device, "type", str(estimator_device).split(":", 1)[0])
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            pc_full, pc_segments, _ = self._estimator.extract_point_clouds(
                depth,
                cam_k,
                segmap=segmap,
                segmap_id=segmap_id,
                skip_border_objects=skip_border_objects,
                z_range=z_range,
            )

            pred_grasps, scores, contact_pts, _ = self._estimator.predict_scene_grasps(
                pc_full,
                pc_segments=pc_segments,
                local_regions=local_regions,
                filter_grasps=filter_grasps,
                forward_passes=forward_passes,
            )

            current_retries = 0
            while len(pred_grasps.get(segmap_id, [])) == 0 and current_retries < max_retries:
                if segmap_id in pc_segments and len(pc_segments[segmap_id]) > 0:
                    target_centroid = np.mean(pc_segments[segmap_id], axis=0)
                else:
                    target_centroid = np.mean(pc_full, axis=0)

                position, wxyz = _sample_random_camera_viewpoint(target_centroid, xy_extent_meters=0.25)
                tf_wc = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position]))
                tf_cw_matrix = tf_wc.inverse().as_matrix()

                pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1))])
                pc_full_cam = (tf_cw_matrix @ pc_full_h.T).T[:, :3]
                pc_segments_cam = {}
                for seg_id, pts in pc_segments.items():
                    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
                    pc_segments_cam[seg_id] = (tf_cw_matrix @ pts_h.T).T[:, :3]

                pred_grasps_new, scores_new, contact_pts_new, _ = self._estimator.predict_scene_grasps(
                    pc_full_cam,
                    pc_segments=pc_segments_cam,
                    local_regions=local_regions,
                    filter_grasps=filter_grasps,
                    forward_passes=forward_passes,
                )

                tf_wc_matrix = tf_wc.as_matrix()
                if segmap_id in pred_grasps_new and len(pred_grasps_new[segmap_id]) > 0:
                    grasps_cam = pred_grasps_new[segmap_id]
                    pred_grasps[segmap_id] = np.matmul(tf_wc_matrix, grasps_cam)
                    pts_cam = contact_pts_new[segmap_id]
                    pts_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
                    contact_pts[segmap_id] = (tf_wc_matrix @ pts_h.T).T[:, :3]
                    scores[segmap_id] = scores_new[segmap_id]

                current_retries += 1

        return GraspNetResult(
            grasps=np.asarray(pred_grasps.get(segmap_id, [])),
            scores=np.asarray(scores.get(segmap_id, [])),
            contact_points=np.asarray(contact_pts.get(segmap_id, [])),
        )

    def plan_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
        segmap_id: int = 1,
        local_regions: bool = True,
        filter_grasps: bool = True,
        forward_passes: int = 1,
    ) -> GraspNetResult:
        """Plan grasps from precomputed point clouds."""
        self.reset()
        pred_grasps, scores, contact_pts, _ = self._estimator.predict_scene_grasps(
            pc_full,
            pc_segments={segmap_id: pc_segment},
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
        )
        return GraspNetResult(
            grasps=np.asarray(pred_grasps.get(segmap_id, [])),
            scores=np.asarray(scores.get(segmap_id, [])),
            contact_points=np.asarray(contact_pts.get(segmap_id, [])),
        )

    def inference(self, **kwargs: Any) -> GraspNetResult:
        """Run local Contact-GraspNet inference.

        Pass either ``depth``/``cam_k``/``segmap`` or ``pc_full``/``pc_segment``.
        """
        if {"depth", "cam_k", "segmap"}.issubset(kwargs):
            return self.plan(**kwargs)
        if {"pc_full", "pc_segment"}.issubset(kwargs):
            return self.plan_point_clouds(**kwargs)
        raise ValueError("Expected depth/cam_k/segmap or pc_full/pc_segment inputs")
