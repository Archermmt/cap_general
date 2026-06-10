"""Local Contact-GraspNet model implementation."""

from dataclasses import dataclass
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cap_general.core.policy.base_policy import PolicyBase


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
    import yaml

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


@PolicyBase.register()
class GraspNetPolicy(PolicyBase):
    """Local Contact-GraspNet grasp planning model."""

    name = "Contact-GraspNet Policy Model"

    def __init__(
        self,
        vendor_root: str | Path,
        checkpoint_root: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_name: str = "model.pt",
        device: str = "cuda",
    ):
        self._vendor_root = Path(vendor_root)
        self._checkpoint_root = (
            Path(checkpoint_root)
            if checkpoint_root is not None
            else self._vendor_root / "checkpoints" / "contact_graspnet"
        )
        self._checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else self._checkpoint_root / "checkpoints"
        )
        self._checkpoint_name = checkpoint_name
        self._device = device
        self._estimator = None

    @classmethod
    def policy_type(cls) -> str:
        return "graspnet"

    def _load_model(self):
        """Lazily load Contact-GraspNet locally."""
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
                "GraspNetPolicy requires contact_graspnet_pytorch. "
                f"Checked vendor_root={self._vendor_root}"
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
        forward_passes: int = 1,
    ) -> GraspNetResult:
        """Plan grasps from depth, camera intrinsics, and segmentation map."""
        self._load_model()
        z_range = [0.2, 2.0] if z_range is None else z_range
        pc_full, pc_segments, _ = self._estimator.extract_point_clouds(
            depth,
            cam_k,
            segmap=segmap,
            segmap_id=segmap_id,
            skip_border_objects=skip_border_objects,
            z_range=z_range,
        )
        return self.plan_point_clouds(
            pc_full=pc_full,
            pc_segment=pc_segments.get(segmap_id, np.empty((0, 3))),
            segmap_id=segmap_id,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
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
        self._load_model()
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

    @property
    def policy_name(self) -> str:
        return "GraspNetPolicy"
