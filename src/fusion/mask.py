"""Sensor valid mask generation."""

from __future__ import annotations

import cv2
import numpy as np

from src.fusion.config import FusionConfig


def build_valid_mask(
    sensor_depth: np.ndarray,
    config: FusionConfig | None = None,
    obj_mask: np.ndarray | None = None,
) -> np.ndarray:
    cfg = config or FusionConfig()
    valid = np.isfinite(sensor_depth) & (sensor_depth > cfg.depth_min) & (sensor_depth < cfg.depth_max)

    if cfg.median_filter_size > 0:
        filled = np.nan_to_num(sensor_depth, nan=0.0).astype(np.float32)
        k = cfg.median_filter_size | 1
        filtered = cv2.medianBlur(filled, k)
        jump = np.abs(filtered - filled) > cfg.ransac_threshold
        valid &= ~jump

    if obj_mask is not None:
        valid &= obj_mask

    return valid
