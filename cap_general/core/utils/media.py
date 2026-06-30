"""Image and video helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any


def frame_to_array(frame: Any):
    """Convert a frame-like object into a uint8 RGB numpy array."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Converting frames requires pillow and numpy") from exc

    if isinstance(frame, bytes):
        frame = Image.open(io.BytesIO(frame))
    if hasattr(frame, "convert"):
        frame = frame.convert("RGB")

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array


def save_video(path: str | Path, frames: list[Any]) -> str:
    """Save frames as a video and return the saved path string."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("Saving video requires imageio") from exc
    target_path = Path(path)
    iio.imwrite(target_path, [frame_to_array(frame) for frame in frames])
    return str(target_path)


def save_image(path: str | Path, image: Any) -> str:
    """Save an image-like object and return the saved path string."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("Saving observation images requires imageio") from exc
    target_path = Path(path)
    iio.imwrite(target_path, frame_to_array(image))
    return str(target_path)
