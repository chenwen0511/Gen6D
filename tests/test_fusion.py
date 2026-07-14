"""Unit tests for depth fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion import fuse_sensor_and_estimated
from src.fusion.align import align_scale_shift, apply_scale_shift
from src.fusion.mask import build_valid_mask


def test_build_valid_mask():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    depth[2:8, 2:8] = 500.0
    depth[0, 0] = 10.0
    depth[9, 9] = 9000.0
    mask = build_valid_mask(depth)
    assert mask[3, 3]
    assert not mask[0, 0]
    assert not mask[9, 9]
    assert mask.sum() == 36


def test_align_scale_shift():
    est = np.linspace(1.0, 3.0, 25, dtype=np.float32).reshape(5, 5)
    sensor = 100.0 * est + 50.0
    mask = np.ones_like(est, dtype=bool)
    result = align_scale_shift(est, sensor, mask)
    assert result.success
    assert abs(result.scale - 100.0) < 1.0
    assert abs(result.shift - 50.0) < 1.0


def test_fuse_sensor_and_estimated():
    h, w = 48, 64
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    sensor = np.full((h, w), np.nan, dtype=np.float32)
    sensor[10:40, 10:50] = 800.0
    yy, xx = np.indices((24, 32))
    est = (1.0 + xx / 32.0 + yy / 24.0).astype(np.float32)
    result = fuse_sensor_and_estimated(sensor, est, rgb)
    assert result.fused_depth.shape == (h, w)
    assert np.isfinite(result.fused_depth).mean() > 0.5
    assert result.scale > 0


def test_apply_scale_shift():
    depth = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    out = apply_scale_shift(depth, 10.0, 5.0)
    assert out[0, 0] == 15.0
    assert np.isnan(out[1, 0])


def test_blend_boundary_skips_invalid_sensor():
    from src.fusion.smooth import blend_boundary

    sensor = np.full((10, 10), np.nan, dtype=np.float32)
    sensor[2:8, 2:8] = 500.0
    metric = np.full((10, 10), 400.0, dtype=np.float32)
    valid = np.isfinite(sensor) & (sensor > 0)
    fused_raw = np.where(valid, sensor, metric).astype(np.float32)

    out = blend_boundary(fused_raw, sensor, metric, valid)
    assert np.isfinite(out).all()
    assert (out[~valid] == 400.0).all()


def test_median_ratio_fallback():
    est = np.linspace(0.5, 2.0, 100, dtype=np.float32).reshape(10, 10)
    sensor = 700.0 * est + np.random.default_rng(0).normal(0, 20, est.shape).astype(np.float32)
    sensor = np.abs(sensor)
    mask = np.ones_like(est, dtype=bool)
    result = align_scale_shift(est, sensor, mask)
    assert result.success
    assert result.scale > 0
    assert abs(result.scale - 700.0) < 100.0


if __name__ == "__main__":
    test_build_valid_mask()
    test_align_scale_shift()
    test_fuse_sensor_and_estimated()
    test_apply_scale_shift()
    print("all fusion tests passed")
