"""Depth map resize utilities."""

from __future__ import annotations

import cv2
import numpy as np


def resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if depth.shape[1] == width and depth.shape[0] == height:
        return depth.astype(np.float32, copy=False)

    resized = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    invalid = ~np.isfinite(depth) | (depth <= 0)
    if invalid.any():
        invalid_f = invalid.astype(np.float32)
        invalid_resized = cv2.resize(invalid_f, (width, height), interpolation=cv2.INTER_LINEAR)
        resized[invalid_resized > 0.5] = np.nan
    return resized.astype(np.float32)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask.astype(bool, copy=False)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
