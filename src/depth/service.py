"""Depth Anything 3 inference service."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image

from src.depth.pointcloud import build_pointcloud_files, scale_intrinsics


DEFAULT_MODEL_DIR = Path("/home/ubuntu/stephen/02-weight/depth-anything/DA3-SMALL")
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/stephen/01-code/Gen6D/outputs/pointclouds")
DEFAULT_MAX_POINTS = 50_000


def load_intrinsics_from_dict(data: dict) -> np.ndarray | None:
    cam_k = data.get("cam_K")
    if cam_k is None:
        return None
    return np.array(cam_k, dtype=np.float32).reshape(3, 3)


def load_intrinsics(camera_json: Path) -> np.ndarray | None:
    if not camera_json.exists():
        return None
    with camera_json.open() as f:
        return load_intrinsics_from_dict(json.load(f))


def load_sensor_depth_from_image(depth: Image.Image, depth_scale: float = 1.0) -> np.ndarray:
    arr = np.array(depth)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32)
    arr[arr == 0] = np.nan
    return arr / depth_scale


def load_sensor_depth(depth_path: Path, depth_scale: float = 1.0) -> np.ndarray | None:
    if not depth_path.exists():
        return None
    return load_sensor_depth_from_image(Image.open(depth_path), depth_scale)


def depth_to_colormap(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
        return Image.fromarray(vis)

    d_min = np.nanpercentile(depth[valid], 5)
    d_max = np.nanpercentile(depth[valid], 95)
    norm = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)

    r = np.clip(1.5 - np.abs(4 * norm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * norm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * norm - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def conf_to_colormap(conf: np.ndarray) -> Image.Image:
    c_min, c_max = float(conf.min()), float(conf.max())
    norm = (conf - c_min) / max(c_max - c_min, 1e-6)
    return depth_to_colormap(norm)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class DepthResult:
    pred_depth: np.ndarray
    pred_conf: np.ndarray
    processed_rgb: Image.Image
    pred_depth_vis: Image.Image
    conf_vis: Image.Image
    depth_range: tuple[float, float]
    conf_range: tuple[float, float]
    intrinsics: np.ndarray | None = None
    sensor_depth_vis: Image.Image | None = None
    sensor_valid_ratio: float | None = None
    sensor_depth_range: tuple[float, float] | None = None
    pointcloud_glb: Path | None = None
    pointcloud_ply: Path | None = None
    point_count: int | None = None

    def to_summary(self) -> dict:
        summary = {
            "depth_shape": list(self.pred_depth.shape),
            "depth_range": list(self.depth_range),
            "conf_range": list(self.conf_range),
        }
        if self.sensor_valid_ratio is not None:
            summary["sensor_valid_ratio"] = self.sensor_valid_ratio
        if self.sensor_depth_range is not None:
            summary["sensor_depth_range"] = list(self.sensor_depth_range)
        if self.point_count is not None:
            summary["point_count"] = self.point_count
        return summary


class DepthService:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        device: str | None = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        max_points: int = DEFAULT_MAX_POINTS,
    ):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.max_points = max_points
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: DepthAnything3 | None = None

    def load_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        self._model = DepthAnything3.from_pretrained(str(self.model_dir))
        self._model = self._model.to(device=torch.device(self.device))

    @property
    def model(self) -> DepthAnything3:
        self.load_model()
        assert self._model is not None
        return self._model

    def predict(
        self,
        rgb: Image.Image | str | Path,
        intrinsics: np.ndarray | None = None,
        sensor_depth: np.ndarray | None = None,
    ) -> DepthResult:
        if isinstance(rgb, (str, Path)):
            rgb_input = str(rgb)
        else:
            rgb_input = rgb

        intrinsics_batch = intrinsics[None] if intrinsics is not None else None

        prediction = self.model.inference(
            [rgb_input if isinstance(rgb_input, str) else rgb_input],
            intrinsics=intrinsics_batch,
        )

        pred_depth = prediction.depth[0]
        pred_conf = prediction.conf[0]
        pred_intrinsics = prediction.intrinsics[0] if prediction.intrinsics is not None else None
        if intrinsics is not None and pred_intrinsics is not None:
            # Prefer model-adjusted intrinsics for processed resolution.
            pc_intrinsics = pred_intrinsics.astype(np.float32)
        elif intrinsics is not None:
            processed_rgb = prediction.processed_images[0]
            if isinstance(rgb_input, str):
                orig_w, orig_h = Image.open(rgb_input).size
            elif isinstance(rgb_input, Image.Image):
                orig_w, orig_h = rgb_input.size
            else:
                orig_w, orig_h = processed_rgb.shape[1], processed_rgb.shape[0]
            pc_intrinsics = scale_intrinsics(
                intrinsics,
                src_size=(orig_w, orig_h),
                dst_size=(pred_depth.shape[1], pred_depth.shape[0]),
            )
        elif pred_intrinsics is not None:
            pc_intrinsics = pred_intrinsics.astype(np.float32)
        else:
            pc_intrinsics = None

        sensor_depth_vis = None
        sensor_valid_ratio = None
        sensor_depth_range = None
        if sensor_depth is not None:
            sensor_depth_vis = depth_to_colormap(sensor_depth)
            valid = np.isfinite(sensor_depth) & (sensor_depth > 0)
            if valid.any():
                sensor_valid_ratio = float(valid.mean())
                sensor_depth_range = (
                    float(np.nanmin(sensor_depth)),
                    float(np.nanmax(sensor_depth)),
                )

        pointcloud_glb = None
        pointcloud_ply = None
        point_count = None
        if pc_intrinsics is not None:
            pc_info = build_pointcloud_files(
                depth=pred_depth,
                rgb=prediction.processed_images[0],
                intrinsics=pc_intrinsics,
                output_dir=self.output_dir,
                max_points=self.max_points,
                conf=pred_conf,
            )
            pointcloud_glb = pc_info["glb_path"]
            pointcloud_ply = pc_info["ply_path"]
            point_count = pc_info["point_count"]

        return DepthResult(
            pred_depth=pred_depth,
            pred_conf=pred_conf,
            processed_rgb=Image.fromarray(prediction.processed_images[0]),
            pred_depth_vis=depth_to_colormap(pred_depth),
            conf_vis=conf_to_colormap(pred_conf),
            depth_range=(float(pred_depth.min()), float(pred_depth.max())),
            conf_range=(float(pred_conf.min()), float(pred_conf.max())),
            intrinsics=pc_intrinsics,
            sensor_depth_vis=sensor_depth_vis,
            sensor_valid_ratio=sensor_valid_ratio,
            sensor_depth_range=sensor_depth_range,
            pointcloud_glb=pointcloud_glb,
            pointcloud_ply=pointcloud_ply,
            point_count=point_count,
        )
