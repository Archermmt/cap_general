"""SAM3 model operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cap_general.core.operator.base_operator import BaseOperator, to_stage_fn
from cap_general.core.operator.model.base_model_op import ModelOp


@dataclass
class Sam3MaskResult:
    mask: Any
    box: list[float]
    score: float
    label: str


@dataclass
class Sam3PointResult:
    masks: Any
    scores: list[float]


@dataclass
class SAM3Config:
    device: str = "cuda"
    checkpoint_path: str | None = None
    load_from_hf: bool = True
    confidence_threshold: float = 0.0
    enable_inst_interactivity: bool = True


@BaseOperator.register()
class SAM3Op(ModelOp):
    """Local SAM3 image segmentation model."""

    op_type = "sam3"
    config_cls = SAM3Config

    def reset(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None
        super().reset()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise ImportError("SAM3Op requires sam3 and torch to be installed.") from exc

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if "cuda" in self._config.device:
                device_idx = int(self._config.device.split(":")[-1]) if ":" in self._config.device else 0
                torch.cuda.set_device(device_idx)

        if self._config.checkpoint_path:
            model = build_sam3_image_model(
                enable_inst_interactivity=self._config.enable_inst_interactivity,
                checkpoint_path=self._config.checkpoint_path,
                load_from_HF=False,
            )
        elif self._config.load_from_hf:
            model = build_sam3_image_model(
                enable_inst_interactivity=self._config.enable_inst_interactivity,
                load_from_HF=True,
            )
        else:
            model = build_sam3_image_model(enable_inst_interactivity=self._config.enable_inst_interactivity)

        if hasattr(model, "to") and self._config.device:
            model = model.to(self._config.device)
        self._model = model
        self._processor = Sam3Processor(
            model, device=self._config.device, confidence_threshold=self._config.confidence_threshold
        )
        self._torch = torch

    @to_stage_fn
    def segment(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._load_model()
        image = inputs["image"]
        text_prompt = inputs["text_prompt"]
        pil_image = self._to_pil(image)
        device_type = "cuda" if "cuda" in self._config.device else "cpu"

        with self._torch.autocast(device_type, dtype=self._torch.bfloat16):
            state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(state=state, prompt=text_prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or boxes is None or scores is None:
            return {"output": []}

        masks_np = self._to_numpy(masks)
        boxes_np = self._to_numpy(boxes)
        scores_np = self._to_numpy(scores)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np.squeeze(1)

        results = [
            Sam3MaskResult(
                mask=masks_np[i] > 0,
                box=boxes_np[i].tolist(),
                score=float(scores_np[i]),
                label=text_prompt,
            )
            for i in range(len(scores_np))
        ]
        return {"output": sorted(results, key=lambda item: item.score, reverse=True)}

    @to_stage_fn
    def segment_point(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._load_model()
        if getattr(self._model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("SAM3 instance interactivity is not enabled")
        image = inputs["image"]
        point_coords = inputs["point_coords"]
        pil_image = self._to_pil(image)
        device_type = "cuda" if "cuda" in self._config.device else "cpu"
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
            return {"output": Sam3PointResult(masks=np.empty((0, 0, 0)), scores=[])}
        sort_idx = np.argsort(scores_np)[::-1]
        return {
            "output": Sam3PointResult(
                masks=masks_np[sort_idx],
                scores=scores_np[sort_idx].astype(float).tolist(),
            )
        }

    @staticmethod
    def _to_pil(image: Any):
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")

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
