"""Guided filter and boundary smoothing."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from src.fusion.config import FusionConfig


def _box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    ksize = 2 * radius + 1
    return cv2.boxFilter(src, ddepth=-1, ksize=(ksize, ksize), normalize=True)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    if guide.ndim == 3:
        guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    guide = guide.astype(np.float32) / 255.0
    src = src.astype(np.float32)
    valid = np.isfinite(src)
    src_fill = np.where(valid, src, 0.0)

    mean_i = _box_filter(guide, radius)
    mean_p = _box_filter(src_fill, radius)
    mean_ip = _box_filter(guide * src_fill, radius)
    mean_ii = _box_filter(guide * guide, radius)

    cov_ip = mean_ip - mean_i * mean_p
    var_i = mean_ii - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)
    filtered = mean_a * guide + mean_b
    filtered[~valid] = np.nan
    return filtered.astype(np.float32)


def build_boundary_mask(valid_mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros_like(valid_mask, dtype=bool)
    structure = np.ones((2 * width + 1, 2 * width + 1), dtype=bool)
    dilated = ndimage.binary_dilation(valid_mask, structure=structure)
    eroded = ndimage.binary_erosion(valid_mask, structure=structure)
    return dilated & ~eroded


def blend_boundary(
    fused_raw: np.ndarray,
    sensor_depth: np.ndarray,
    metric_est_depth: np.ndarray,
    valid_mask: np.ndarray,
    config: FusionConfig | None = None,
) -> np.ndarray:
    cfg = config or FusionConfig()
    boundary = build_boundary_mask(valid_mask, cfg.boundary_width)
    if not boundary.any():
        return fused_raw

    out = fused_raw.copy()
    dist = ndimage.distance_transform_edt(~valid_mask)
    max_dist = max(float(dist[boundary].max()), 1.0)
    alpha = np.clip(dist / max_dist, 0.0, 1.0)
    # 传感器无效处不能参与混合，否则 alpha*NaN 会把估计深度也污染成 NaN
    sensor_safe = np.where(valid_mask, sensor_depth, metric_est_depth)
    blend = alpha * sensor_safe + (1.0 - alpha) * metric_est_depth
    out[boundary] = blend[boundary]
    return out


def smooth_fused_depth(
    fused_raw: np.ndarray,
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    config: FusionConfig | None = None,
) -> np.ndarray:
    cfg = config or FusionConfig()
    if not cfg.enable_guided_smooth:
        return fused_raw

    inpaint_mask = ~valid_mask & np.isfinite(fused_raw) & (fused_raw > 0)
    if not inpaint_mask.any():
        return fused_raw

    filled = fused_raw.copy()
    boundary = build_boundary_mask(valid_mask, cfg.boundary_width)
    smooth_mask = inpaint_mask | boundary
    # Fill holes only for filtering reference, do not overwrite sensor pixels
    fill_value = float(np.nanmedian(filled[valid_mask])) if valid_mask.any() else 0.0
    filled[~np.isfinite(filled)] = fill_value

    smoothed = guided_filter(rgb, filled, radius=cfg.guided_radius, eps=cfg.guided_eps)

    out = fused_raw.copy()
    replace = smooth_mask & np.isfinite(smoothed) & (smoothed > 0)
    out[replace] = smoothed[replace]
    out[valid_mask] = fused_raw[valid_mask]
    return out
