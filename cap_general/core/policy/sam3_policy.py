"""Local SAM3 segmentation model implementation."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cap_general.core.policy.base_policy import BasePolicy

from .base_policy import BasePolicyConfig


@dataclass
class Sam3MaskResult:
    """One SAM3 text-prompt segmentation result."""

    mask: np.ndarray
    box: list[float]
    score: float
    label: str


@dataclass
class Sam3PointResult:
    """SAM3 point-prompt segmentation result."""

    masks: np.ndarray
    scores: list[float]


@dataclass
class SAM3PolicyConfig(BasePolicyConfig):
    """Configuration for SAM3Policy."""

    device: str = "cuda"
    checkpoint_path: str | None = None
    load_from_hf: bool = True
    confidence_threshold: float = 0.0
    enable_inst_interactivity: bool = True
    describe: str = field(
        default=(
            "Segments image regions from text prompts or point prompts using a local SAM3 image segmentation model."
        ),
        kw_only=True,
    )


@BasePolicy.register()
class SAM3Policy(BasePolicy):
    """Local SAM3 image segmentation model."""

    name = "SAM3 Policy"
    config_cls = SAM3PolicyConfig

    def __init__(self, config: SAM3PolicyConfig, logger: logging.Logger):
        super().__init__(config=config, logger=logger)
        self._device = config.device
        self._checkpoint_path = config.checkpoint_path
        self._load_from_hf = config.load_from_hf
        self._confidence_threshold = config.confidence_threshold
        self._enable_inst_interactivity = config.enable_inst_interactivity
        self._model = None
        self._processor = None
        self._torch = None

    @classmethod
    def policy_type(cls) -> str:
        return "sam3"

    def reset(self) -> None:
        """Load SAM3 locally if needed."""
        if self._model is not None:
            return

        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise ImportError("SAM3Policy requires sam3 and torch to be installed.") from exc

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if "cuda" in self._device:
                device_idx = int(self._device.split(":")[-1]) if ":" in self._device else 0
                torch.cuda.set_device(device_idx)

        if self._checkpoint_path:
            self._model = build_sam3_image_model(
                enable_inst_interactivity=self._enable_inst_interactivity,
                checkpoint_path=self._checkpoint_path,
                load_from_HF=False,
            )
        elif self._load_from_hf:
            self._model = build_sam3_image_model(
                enable_inst_interactivity=self._enable_inst_interactivity,
                load_from_HF=True,
            )
        else:
            self._model = build_sam3_image_model(enable_inst_interactivity=self._enable_inst_interactivity)

        if hasattr(self._model, "to") and self._device:
            self._model = self._model.to(self._device)
        self._processor = Sam3Processor(
            self._model, device=self._device, confidence_threshold=self._confidence_threshold
        )
        self._torch = torch

    @staticmethod
    def _to_pil(image: Any):
        """Convert PIL/numpy image inputs to PIL RGB."""
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")

    def segment(self, image: Any, text_prompt: str) -> list[Sam3MaskResult]:
        """Run local text-prompt segmentation."""
        self.reset()
        pil_image = self._to_pil(image)
        device_type = "cuda" if "cuda" in self._device else "cpu"

        with self._torch.autocast(device_type, dtype=self._torch.bfloat16):
            state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(state=state, prompt=text_prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or boxes is None or scores is None:
            return []

        masks_np = self._to_numpy(masks)
        boxes_np = self._to_numpy(boxes)
        scores_np = self._to_numpy(scores)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np.squeeze(1)

        results = [
            Sam3MaskResult(mask=masks_np[i] > 0, box=boxes_np[i].tolist(), score=float(scores_np[i]), label=text_prompt)
            for i in range(len(scores_np))
        ]
        return sorted(results, key=lambda item: item.score, reverse=True)

    def segment_point(self, image: Any, point_coords: tuple[float, float]) -> Sam3PointResult:
        """Run local point-prompt segmentation."""
        self.reset()
        if getattr(self._model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("SAM3 instance interactivity is not enabled")

        pil_image = self._to_pil(image)
        device_type = "cuda" if "cuda" in self._device else "cpu"
        with self._torch.autocast(device_type, dtype=self._torch.bfloat16):
            state = self._processor.set_image(pil_image)
            point_array = np.array([list(point_coords)], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int64)
            masks, scores, _ = self._model.predict_inst(
                state, point_coords=point_array, point_labels=point_labels, multimask_output=True
            )

        masks_np = np.asarray(masks)
        scores_np = np.asarray(scores)
        if masks_np.size == 0 or scores_np.size == 0:
            return Sam3PointResult(masks=np.empty((0, 0, 0)), scores=[])

        sort_idx = np.argsort(scores_np)[::-1]
        return Sam3PointResult(
            masks=masks_np[sort_idx],
            scores=scores_np[sort_idx].astype(float).tolist(),
        )

    def inference(self, image: Any, text_prompt: str, **_: Any) -> list[Sam3MaskResult]:
        """Run local SAM3 text-prompt segmentation."""
        return self.segment(image=image, text_prompt=text_prompt)

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
            if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
                tensor = tensor.float()
            return tensor.numpy()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
            if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
                tensor = tensor.float()
            return tensor.numpy()
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
        return np.asarray(tensor)
