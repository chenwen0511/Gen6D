"""Scale and shift alignment between estimated and sensor depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.fusion.config import FusionConfig


@dataclass
class AlignResult:
    scale: float
    shift: float
    inlier_ratio: float
    success: bool


def _fit_least_squares(est: np.ndarray, sensor: np.ndarray) -> tuple[float, float]:
    a = np.stack([est, np.ones_like(est)], axis=-1)
    params, _, _, _ = np.linalg.lstsq(a, sensor, rcond=None)
    return float(params[0]), float(params[1])


def align_scale_shift(
    est_depth: np.ndarray,
    sensor_depth: np.ndarray,
    align_mask: np.ndarray,
    config: FusionConfig | None = None,
) -> AlignResult:
    cfg = config or FusionConfig()
    est = est_depth[align_mask]
    sensor = sensor_depth[align_mask]
    finite = np.isfinite(est) & np.isfinite(sensor) & (est > 0) & (sensor > 0)
    est = est[finite]
    sensor = sensor[finite]

    if est.size < 10:
        return AlignResult(scale=1.0, shift=0.0, inlier_ratio=0.0, success=False)

    best_inliers = None
    best_count = -1
    best_scale, best_shift = 1.0, 0.0
    rng = np.random.default_rng(42)

    for _ in range(cfg.ransac_iterations):
        idx = rng.choice(est.size, size=2, replace=False)
        x1, x2 = est[idx[0]], est[idx[1]]
        y1, y2 = sensor[idx[0]], sensor[idx[1]]
        if abs(x2 - x1) < 1e-6:
            continue
        scale = (y2 - y1) / (x2 - x1)
        shift = y1 - scale * x1
        pred = scale * est + shift
        inliers = np.abs(pred - sensor) < cfg.ransac_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_scale, best_shift = scale, shift

    inlier_ratio = best_count / est.size if est.size else 0.0
    if best_inliers is None or inlier_ratio < cfg.min_align_inlier_ratio:
        return AlignResult(scale=1.0, shift=0.0, inlier_ratio=inlier_ratio, success=False)

    scale, shift = _fit_least_squares(est[best_inliers], sensor[best_inliers])
    if scale <= 0:
        return AlignResult(scale=scale, shift=shift, inlier_ratio=inlier_ratio, success=False)

    pred = scale * est + shift
    final_inliers = np.abs(pred - sensor) < cfg.ransac_threshold
    final_ratio = float(final_inliers.mean())
    if final_ratio < cfg.min_align_inlier_ratio:
        return AlignResult(scale=scale, shift=shift, inlier_ratio=final_ratio, success=False)

    return AlignResult(scale=scale, shift=shift, inlier_ratio=final_ratio, success=True)


def apply_scale_shift(depth: np.ndarray, scale: float, shift: float) -> np.ndarray:
    out = scale * depth + shift
    invalid = ~np.isfinite(depth) | (depth <= 0)
    out[invalid] = np.nan
    return out.astype(np.float32)
