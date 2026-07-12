"""Fusion result types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionResult:
    fused_depth: np.ndarray
    metric_est_depth: np.ndarray
    valid_mask: np.ndarray
    inpaint_mask: np.ndarray
    scale: float
    shift: float
    align_inlier_ratio: float
    status: str
    message: str = ""

    def to_summary(self) -> dict:
        valid_ratio = float(self.valid_mask.mean())
        fused_valid = np.isfinite(self.fused_depth) & (self.fused_depth > 0)
        return {
            "status": self.status,
            "message": self.message,
            "scale": self.scale,
            "shift": self.shift,
            "align_inlier_ratio": self.align_inlier_ratio,
            "sensor_valid_ratio": valid_ratio,
            "fused_valid_ratio": float(fused_valid.mean()) if fused_valid.size else 0.0,
            "inpaint_ratio": float(self.inpaint_mask.mean()),
        }
