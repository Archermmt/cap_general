"""Utility helpers for converting tensor/array values to plain Python types."""

from __future__ import annotations

from typing import Any

import numpy as np


def tensor_to_list(value: Any) -> Any:
    """Convert a tensor or ndarray to a nested Python list; pass through others."""
    if value is None:
        return None
    try:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def tensor_to_image_array(value: Any) -> np.ndarray:
    """Convert a tensor or array to a uint8 HxWxC numpy array."""
    try:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value)
    if array.dtype != np.uint8:
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def tensor_to_scalar(value: Any) -> int | float | str | bool | None:
    """Reduce a tensor/array to a scalar Python value, or return None."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            try:
                value = value.mean().item()
            except Exception:
                pass
            if isinstance(value, (bool, int, float)):
                return value
            return None
    except ImportError:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (bool, int, float, str)):
        return value
    return None


def tensor_mean_value(values: Any) -> float | None:
    """Return the mean of a collection of scalars/tensors, or None if not numeric."""
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        numeric = [tensor_to_scalar(v) for v in values]
        numeric = [float(v) for v in numeric if isinstance(v, (int, float))]
        if not numeric:
            return None
        return sum(numeric) / len(numeric)
    scalar = tensor_to_scalar(values)
    if isinstance(scalar, (int, float)):
        return float(scalar)
    return None
