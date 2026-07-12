"""Depth fusion configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionConfig:
    depth_min: float = 50.0
    depth_max: float = 5000.0
    median_filter_size: int = 0
    conf_percentile: float = 20.0
    ransac_iterations: int = 1000
    ransac_threshold: float = 50.0
    min_align_inlier_ratio: float = 0.30
    min_valid_sensor_ratio: float = 0.05
    boundary_width: int = 5
    guided_radius: int = 8
    guided_eps: float = 1e-2
    enable_boundary_blend: bool = True
    enable_guided_smooth: bool = False
