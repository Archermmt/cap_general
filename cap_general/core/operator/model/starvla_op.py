"""StarVLA model operator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cap_general.core.operator.base_operator import BaseOperator, to_stage_fn
from cap_general.core.operator.model.base_model_op import ModelOp


@dataclass
class StarVLAConfig:
    ckpt_path: str = ""
    device: str = "cuda"
    use_bf16: bool = False
    action_dtype: str | None = None
    unnorm_key: str | None = None
    use_ddim: bool = True
    num_ddim_steps: int = 10
    image_size: tuple | list | None = field(default_factory=lambda: [224, 224])


@BaseOperator.register()
class StarVLAOp(ModelOp):
    """Local in-process StarVLA action policy."""

    op_type = "starvla"
    config_cls = StarVLAConfig

    def reset(self) -> None:
        self._image_size = tuple(self._config.image_size) if self._config.image_size else None
        self._framework = None
        self._norm_processor = None
        self._action_chunk_size = 1
        self._task_description: str | None = None
        self._raw_actions = None
        self._device = self._config.device
        self._unnorm_key = self._config.unnorm_key
        super().reset()

    def _load_framework(self, task_description: str | None = None) -> None:
        if self._framework is None or self._norm_processor is None:
            try:
                import torch
                from deployment.model_server.policy_norm_processor import PolicyNormProcessor  # type: ignore
                from starVLA.model.framework.base_framework import baseframework  # type: ignore
                from starVLA.model.framework.share_tools import read_mode_config  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    f"StarVLAOp requires starVLA, deployment, torch. Missing: {getattr(exc, 'name', None)!r}."
                ) from exc
            framework = baseframework.from_pretrained(self._config.ckpt_path)
            device = self._resolve_device(torch)
            if self._config.use_bf16 and device.startswith("cuda"):
                framework = framework.to(torch.bfloat16)
            elif self._config.use_bf16:
                self._logger.warning("Skipping bfloat16 conversion because device=%s is not CUDA", device)
            self._framework = framework.to(device).eval()
            self._patch_action_model_input_dtype(torch)
            self._device = device
            model_cfg, _ = read_mode_config(self._config.ckpt_path)
            action_model_cfg = model_cfg["framework"]["action_model"]
            if "action_horizon" in action_model_cfg:
                self._action_chunk_size = int(action_model_cfg["action_horizon"])
            elif "future_action_window_size" in action_model_cfg:
                self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
            else:
                raise ValueError("StarVLAOp checkpoint config has no action_horizon or future_action_window_size")
            self._norm_processor = PolicyNormProcessor(self._config.ckpt_path, unnorm_key=self._unnorm_key)
            if self._unnorm_key is None:
                self._unnorm_key = self._norm_processor.unnorm_key
        self._task_description = task_description
        self._raw_actions = None

    def _resolve_device(self, torch: Any) -> str:
        requested = (self._device or "auto").lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._logger.warning("CUDA requested but unavailable; using MPS")
                return "mps"
            self._logger.warning("CUDA requested but unavailable; using CPU")
            return "cpu"
        return self._device

    def _patch_action_model_input_dtype(self, torch: Any) -> None:
        if self._config.action_dtype is None:
            return
        action_model = getattr(self._framework, "action_model", None)
        if action_model is None or getattr(action_model, "_cap_dtype_patched", False):
            return
        dtype = getattr(torch, self._config.action_dtype, None)
        if dtype is None:
            raise ValueError(f"Unknown StarVLAOp action_dtype: {self._config.action_dtype!r}")
        predict_action = action_model.predict_action

        def _predict_action_with_dtype(actions_hidden_states, *args, **kwargs):
            return predict_action(actions_hidden_states.to(dtype=dtype), *args, **kwargs)

        action_model.predict_action = _predict_action_with_dtype
        action_model._cap_dtype_patched = True

    @to_stage_fn
    def inference(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._framework is None or self._norm_processor is None:
            self._load_framework()
        example = inputs.get("example")
        image = inputs.get("image")
        images = inputs.get("images")
        lang = inputs.get("lang")
        task_description = inputs.get("task_description")
        step = inputs.get("step", 0)
        predict_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ("example", "image", "images", "lang", "task_description", "step")
        }
        example = self._build_example(
            example=example,
            image=image,
            images=images,
            lang=lang,
            task_description=task_description,
        )
        current_task = example.get("lang")
        if current_task != self._task_description:
            self._load_framework(current_task)
        example = self._resize_example_images(example)
        if step % self._action_chunk_size == 0 or self._raw_actions is None:
            output = self._framework.predict_action(
                examples=[example],
                do_sample=False,
                use_ddim=self._config.use_ddim,
                num_ddim_steps=self._config.num_ddim_steps,
                **predict_kwargs,
            )
            normalized = np.asarray(output["normalized_actions"])
            actions = np.stack(
                [self._norm_processor.unapply_actions(normalized[i]) for i in range(normalized.shape[0])],
                axis=0,
            )
            self._raw_actions = actions[0]
        frame_idx = step % self._action_chunk_size
        raw_action = self._raw_actions[frame_idx]
        return {
            "raw_action": {
                "world_vector": raw_action[:3].copy(),
                "rotation_delta": raw_action[3:6].copy(),
                "open_gripper": raw_action[6:7].copy(),
            }
        }

    def _build_example(self, example, image, images, lang, task_description) -> dict[str, Any]:
        if example is not None:
            return dict(example)
        resolved_images = images or ([image] if image is not None else None)
        if resolved_images is None:
            raise ValueError("StarVLAOp inference requires example, image, or images")
        resolved_lang = lang if lang is not None else task_description
        if resolved_lang is None:
            raise ValueError("StarVLAOp inference requires lang or task_description")
        return {"image": resolved_images, "lang": resolved_lang}

    def _resize_example_images(self, example: dict[str, Any]) -> dict[str, Any]:
        if not self._image_size or not example.get("image"):
            return example
        from PIL import Image

        target_height, target_width = self._image_size
        resized = []
        for img in example["image"]:
            array = np.asarray(img)
            if array.shape[:2] != (target_height, target_width):
                array = np.asarray(
                    Image.fromarray(array).resize((target_width, target_height), Image.BILINEAR)
                )
            resized.append(array)
        return {**example, "image": resized}
