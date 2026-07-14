"""Depth fusion pipeline."""

from __future__ import annotations

import numpy as np

from src.fusion.align import align_scale_shift, apply_scale_shift
from src.fusion.config import FusionConfig
from src.fusion.mask import build_valid_mask
from src.fusion.resize import resize_depth, resize_mask, resize_rgb
from src.fusion.smooth import blend_boundary, smooth_fused_depth
from src.fusion.types import FusionResult


def fuse_sensor_and_estimated(
    sensor_depth: np.ndarray,
    est_depth: np.ndarray,
    rgb: np.ndarray,
    conf: np.ndarray | None = None,
    obj_mask: np.ndarray | None = None,
    config: FusionConfig | None = None,
) -> FusionResult:
    cfg = config or FusionConfig()
    target_size = (sensor_depth.shape[1], sensor_depth.shape[0])

    est_resized = resize_depth(est_depth, target_size)
    rgb_resized = resize_rgb(rgb, target_size)
    conf_resized = resize_depth(conf, target_size) if conf is not None else None
    obj_mask_resized = resize_mask(obj_mask, target_size) if obj_mask is not None else None

    valid_mask = build_valid_mask(sensor_depth, cfg, obj_mask_resized)
    valid_ratio = float(valid_mask.mean())

    if valid_ratio < cfg.min_valid_sensor_ratio:
        align = align_scale_shift(est_resized, sensor_depth, valid_mask, cfg)
        if align.success:
            metric_est = apply_scale_shift(est_resized, align.scale, align.shift)
            return FusionResult(
                fused_depth=metric_est,
                metric_est_depth=metric_est,
                valid_mask=valid_mask,
                inpaint_mask=~valid_mask,
                scale=align.scale,
                shift=align.shift,
                align_inlier_ratio=align.inlier_ratio,
                status="fallback_metric_est",
                message="传感器有效点过少，仅输出对齐后的估计深度",
            )
        return FusionResult(
            fused_depth=est_resized,
            metric_est_depth=est_resized,
            valid_mask=valid_mask,
            inpaint_mask=~valid_mask,
            scale=1.0,
            shift=0.0,
            align_inlier_ratio=0.0,
            status="fallback_raw_est",
            message="传感器有效点过少且对齐失败，输出原始估计深度",
        )

    align_mask = valid_mask.copy()
    if conf_resized is not None and align_mask.any():
        thresh = np.percentile(conf_resized[align_mask], cfg.conf_percentile)
        align_mask &= conf_resized >= thresh

    if align_mask.sum() < 10:
        align_mask = valid_mask

    align = align_scale_shift(est_resized, sensor_depth, align_mask, cfg)
    if not align.success:
        scale, shift = align.scale, align.shift
        if scale <= 0:
            scale, shift = 1.0, 0.0
        metric_est = apply_scale_shift(est_resized, scale, shift)
        metric_est[~np.isfinite(metric_est) | (metric_est <= 0)] = np.nan
        fused_raw = np.where(valid_mask, sensor_depth, metric_est).astype(np.float32)
        if cfg.enable_boundary_blend:
            fused_raw = blend_boundary(fused_raw, sensor_depth, metric_est, valid_mask, cfg)
        fused = smooth_fused_depth(fused_raw, rgb_resized, valid_mask, cfg)
        fused[~np.isfinite(fused) | (fused <= 0)] = np.nan
        return FusionResult(
            fused_depth=fused,
            metric_est_depth=metric_est,
            valid_mask=valid_mask,
            inpaint_mask=~valid_mask & np.isfinite(fused) & (fused > 0),
            scale=scale,
            shift=shift,
            align_inlier_ratio=align.inlier_ratio,
            status="fallback_inpaint_only",
            message="对齐未完全成功，已用估计尺度补洞并保留传感器有效区",
        )

    metric_est = apply_scale_shift(est_resized, align.scale, align.shift)
    metric_est[~np.isfinite(metric_est) | (metric_est <= 0)] = np.nan
    fused_raw = np.where(valid_mask, sensor_depth, metric_est).astype(np.float32)

    if cfg.enable_boundary_blend:
        fused_raw = blend_boundary(fused_raw, sensor_depth, metric_est, valid_mask, cfg)

    fused = smooth_fused_depth(fused_raw, rgb_resized, valid_mask, cfg)
    fused[~np.isfinite(fused) | (fused <= 0)] = np.nan
    inpaint_mask = ~valid_mask & np.isfinite(fused) & (fused > 0)

    return FusionResult(
        fused_depth=fused,
        metric_est_depth=metric_est,
        valid_mask=valid_mask,
        inpaint_mask=inpaint_mask,
        scale=align.scale,
        shift=align.shift,
        align_inlier_ratio=align.inlier_ratio,
        status="success",
        message="深度融合完成",
    )
