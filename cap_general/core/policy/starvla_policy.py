"""StarVLA policy implementation."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cap_general.core.policy.base_policy import BasePolicy

from .base_policy import BasePolicyConfig


@dataclass
class StarVLAPolicyConfig(BasePolicyConfig):
    """Configuration for StarVLAPolicy."""

    ckpt_path: str
    device: str = "cuda"
    use_bf16: bool = False
    unnorm_key: str | None = None
    use_ddim: bool = True
    num_ddim_steps: int = 10
    image_size: tuple[int, int] | list[int] | None = field(default_factory=lambda: [224, 224])
    describe: str = field(
        default=(
            "Predicts robot end-effector delta actions from camera images and "
            "language task descriptions with StarVLA."
        ),
        kw_only=True,
    )


@BasePolicy.register()
class StarVLAPolicy(BasePolicy):
    """Local in-process StarVLA action policy."""

    config_cls = StarVLAPolicyConfig

    def __init__(
        self,
        config: StarVLAPolicyConfig,
        logger: logging.Logger | None = None,
    ):
        super().__init__(config=config, logger=logger)
        self._ckpt_path = config.ckpt_path
        self._device = config.device
        self._use_bf16 = config.use_bf16
        self._unnorm_key = config.unnorm_key
        self._use_ddim = config.use_ddim
        self._num_ddim_steps = config.num_ddim_steps
        self._image_size = tuple(config.image_size) if config.image_size else None

        self._framework = None
        self._norm_processor = None
        self._action_chunk_size = 1
        self._task_description: str | None = None
        self._raw_actions = None

    @classmethod
    def policy_type(cls) -> str:
        return "starvla"

    def _load_model(self) -> None:
        """Lazily load StarVLA and its action normalization processor."""
        if self._framework is not None:
            return

        try:
            import torch
            from deployment.model_server.policy_norm_processor import (  # type: ignore[import-not-found]
                PolicyNormProcessor,
            )
            from starVLA.model.framework.base_framework import (  # type: ignore[import-not-found]
                baseframework,
            )
            from starVLA.model.framework.share_tools import (  # type: ignore[import-not-found]
                read_mode_config,
            )
        except ImportError as exc:
            missing_module = getattr(exc, "name", None)
            raise ImportError(
                "StarVLAPolicy requires starVLA, deployment, torch, pillow, and numpy "
                f"to be importable in the current environment. Missing module: {missing_module!r}."
                " Install StarVLA into the active Python environment."
            ) from exc

        framework = baseframework.from_pretrained(self._ckpt_path)
        device = self._resolve_device(torch)
        if self._use_bf16 and device.startswith("cuda"):
            framework = framework.to(torch.bfloat16)
        elif self._use_bf16:
            self.logger.warning("Skipping bfloat16 conversion because device=%s is not CUDA", device)
        self._framework = framework.to(device).eval()
        self._device = device

        model_cfg, _ = read_mode_config(self._ckpt_path)
        action_model_cfg = model_cfg["framework"]["action_model"]
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                "StarVLAPolicy checkpoint config has no action_horizon or future_action_window_size"
            )

        self._norm_processor = PolicyNormProcessor(
            self._ckpt_path,
            unnorm_key=self._unnorm_key,
        )
        if self._unnorm_key is None:
            self._unnorm_key = self._norm_processor.unnorm_key

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
                self.logger.warning("CUDA requested but unavailable; using MPS on this machine")
                return "mps"
            self.logger.warning("CUDA requested but unavailable; using CPU")
            return "cpu"
        return self._device

    def reset(self, task_description: str | None = None) -> None:
        """Reset cached episode-level action chunks."""
        self._task_description = task_description
        self._raw_actions = None

    def predict_action(
        self,
        examples: list[dict[str, Any]],
        unnorm_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Predict and unnormalize StarVLA action chunks."""
        self._load_model()
        output = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(output["normalized_actions"])
        actions = np.stack(
            [
                self._norm_processor.unapply_actions(normalized[batch_idx])
                for batch_idx in range(normalized.shape[0])
            ],
            axis=0,
        )
        return {
            "actions": actions,
            "unnorm_key": unnorm_key or self._unnorm_key,
        }

    def inference(
        self,
        example: dict[str, Any] | None = None,
        image: Any | None = None,
        images: list[Any] | None = None,
        lang: str | None = None,
        task_description: str | None = None,
        step: int = 0,
        **predict_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one StarVLA policy step and return an action dictionary."""
        self._load_model()
        example = self._build_example(
            example=example,
            image=image,
            images=images,
            lang=lang,
            task_description=task_description,
        )
        current_task = example.get("lang")
        if current_task != self._task_description:
            self.reset(current_task)

        example = self._resize_example_images(example)
        if step % self._action_chunk_size == 0 or self._raw_actions is None:
            result = self.predict_action(
                examples=[example],
                do_sample=False,
                use_ddim=self._use_ddim,
                num_ddim_steps=self._num_ddim_steps,
                **predict_kwargs,
            )
            self._raw_actions = np.asarray(result["actions"])[0]

        frame_idx = step % self._action_chunk_size
        raw_action = self._raw_actions[frame_idx]
        return {
            "raw_action": {
                "world_vector": raw_action[:3].copy(),
                "rotation_delta": raw_action[3:6].copy(),
                "open_gripper": raw_action[6:7].copy(),
            }
        }

    def _build_example(
        self,
        example: dict[str, Any] | None,
        image: Any | None,
        images: list[Any] | None,
        lang: str | None,
        task_description: str | None,
    ) -> dict[str, Any]:
        if example is not None:
            return dict(example)

        resolved_images = images
        if resolved_images is None and image is not None:
            resolved_images = [image]
        if resolved_images is None:
            raise ValueError("StarVLAPolicy inference requires example, image, or images")

        resolved_lang = lang if lang is not None else task_description
        if resolved_lang is None:
            raise ValueError("StarVLAPolicy inference requires lang or task_description")
        return {"image": resolved_images, "lang": resolved_lang}

    def _resize_example_images(self, example: dict[str, Any]) -> dict[str, Any]:
        if not self._image_size or not example.get("image"):
            return example

        from PIL import Image

        target_height, target_width = self._image_size
        resized_images = []
        for image in example["image"]:
            array = np.asarray(image)
            if array.shape[:2] != (target_height, target_width):
                array = np.asarray(
                    Image.fromarray(array).resize((target_width, target_height), Image.BILINEAR)
                )
            resized_images.append(array)
        return {**example, "image": resized_images}
