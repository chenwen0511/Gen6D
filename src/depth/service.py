"""Depth Anything 3 inference service."""

from __future__ import annotations

import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image, ImageDraw

from src.depth.pointcloud import build_pointcloud_files, resolve_rgb_shift_xy
from src.fusion import fuse_sensor_and_estimated
from src.fusion.types import FusionResult
from src.storage.upload_archive import DEFAULT_SESSION_DIR, archive_session


DEFAULT_MODEL_DIR = Path("/home/ubuntu/stephen/02-weight/depth-anything/DA3-SMALL")
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/stephen/01-code/Gen6D/outputs/pointclouds")
DEFAULT_MAX_POINTS = 50_000


def normalize_camera_json(data: object) -> dict:
    """
    统一 camera.json：支持 ``{"cam_K":[...],"depth_scale":...}``，
    或直接 9 元 / 3×3 数组（视为 cam_K）。
    """
    if isinstance(data, dict):
        return data
    if isinstance(data, (list, tuple)):
        flat = np.asarray(data, dtype=np.float64).reshape(-1)
        if flat.size == 9:
            return {"cam_K": flat.tolist(), "depth_scale": 1.0}
        raise ValueError(f"camera.json 数组长度应为 9，当前 {flat.size}")
    raise ValueError(f"camera.json 应为对象或 9 元数组，当前类型 {type(data).__name__}")


def load_intrinsics_from_dict(data: dict | list | tuple | None) -> np.ndarray | None:
    if data is None:
        return None
    payload = normalize_camera_json(data)
    cam_k = payload.get("cam_K")
    if cam_k is None:
        return None
    return np.array(cam_k, dtype=np.float32).reshape(3, 3)


def load_intrinsics(camera_json: Path) -> np.ndarray | None:
    if not camera_json.exists():
        return None
    with camera_json.open() as f:
        return load_intrinsics_from_dict(json.load(f))


def raw_depth_to_mm(arr: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert raw depth units to millimeters.

    - ``depth_scale >= 0.1`` (e.g. 1.0): legacy Gen6D, ``depth_mm = raw / depth_scale``
    - ``depth_scale < 0.1`` (e.g. 0.001): RealSense export, ``depth_mm = raw * depth_scale * 1000``
    """
    if depth_scale > 0 and depth_scale < 0.1:
        return arr * depth_scale * 1000.0
    return arr / depth_scale


def load_sensor_depth_from_image(depth: Image.Image, depth_scale: float = 1.0) -> np.ndarray:
    arr = np.array(depth)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32)
    arr[arr == 0] = np.nan
    return raw_depth_to_mm(arr, depth_scale)


def load_sensor_depth(depth_path: Path, depth_scale: float = 1.0) -> np.ndarray | None:
    if not depth_path.exists():
        return None
    return load_sensor_depth_from_image(Image.open(depth_path), depth_scale)


def _apply_depth_colormap(norm: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * norm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * norm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * norm - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _depth_value_range(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    d_min = float(np.nanpercentile(depth[valid], 5))
    d_max = float(np.nanpercentile(depth[valid], 95))
    return d_min, d_max


def _format_depth_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def append_depth_colorbar(
    image: Image.Image,
    d_min: float,
    d_max: float,
    unit: str = "mm",
    bar_width: int = 64,
) -> Image.Image:
    img = image.convert("RGB")
    h, w = img.height, img.width
    canvas = Image.new("RGB", (w + bar_width + 16, h), (245, 245, 245))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    bar_x0 = w + 14
    bar_x1 = w + bar_width - 6
    bar_top, bar_bottom = 18, h - 18
    bar_h = max(bar_bottom - bar_top, 1)

    for i in range(bar_h):
        t = 1.0 - i / max(bar_h - 1, 1)
        color = tuple(_apply_depth_colormap(np.array([t]))[0].tolist())
        y = bar_top + i
        draw.line([(bar_x0, y), (bar_x1, y)], fill=color, width=1)

    draw.rectangle([(bar_x0, bar_top), (bar_x1, bar_bottom)], outline=(60, 60, 60), width=1)
    draw.text((bar_x0, 4), _format_depth_value(d_max), fill=(20, 20, 20))
    draw.text((bar_x0, bar_bottom + 2), _format_depth_value(d_min), fill=(20, 20, 20))
    draw.text((bar_x0, bar_top + bar_h // 2 - 6), "远", fill=(180, 40, 40))
    draw.text((bar_x0, bar_bottom - 22), "近", fill=(40, 80, 200))
    if unit:
        draw.text((w + 8, h - 14), unit, fill=(80, 80, 80))

    return canvas


def depth_to_colormap(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
        return Image.fromarray(vis)

    d_min, d_max = _depth_value_range(depth, valid)
    norm = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    rgb = _apply_depth_colormap(norm)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def depth_to_colormap_with_colorbar(
    depth: np.ndarray,
    unit: str = "mm",
) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return append_depth_colorbar(
            Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8)),
            0.0,
            0.0,
            unit=unit,
        )

    d_min, d_max = _depth_value_range(depth, valid)
    base = depth_to_colormap(depth)
    return append_depth_colorbar(base, d_min, d_max, unit=unit)


def conf_to_colormap(conf: np.ndarray) -> Image.Image:
    c_min, c_max = float(conf.min()), float(conf.max())
    norm = (conf - c_min) / max(c_max - c_min, 1e-6)
    return depth_to_colormap(norm)


def fusion_region_to_vis(valid_mask: np.ndarray, inpaint_mask: np.ndarray) -> Image.Image:
    vis = np.zeros((*valid_mask.shape, 3), dtype=np.uint8)
    vis[valid_mask] = [60, 160, 60]
    vis[inpaint_mask] = [255, 160, 40]
    return Image.fromarray(vis)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _load_rgb_array(rgb: Image.Image | str | Path) -> np.ndarray:
    if isinstance(rgb, (str, Path)):
        return np.array(Image.open(rgb).convert("RGB"))
    return np.array(rgb.convert("RGB"))


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
    fused_depth: np.ndarray | None = None
    fused_depth_vis: Image.Image | None = None
    fusion_mask_vis: Image.Image | None = None
    fusion: dict | None = None
    pointcloud_glb: Path | None = None
    pointcloud_ply: Path | None = None
    point_count: int | None = None
    sensor_pointcloud_glb: Path | None = None
    sensor_pointcloud_ply: Path | None = None
    sensor_point_count: int | None = None
    session_dir: Path | None = None
    rgb_shift: dict | None = None

    def to_summary(self) -> dict:
        summary = {
            "depth_shape": list(self.pred_depth.shape),
            "depth_range": list(self.depth_range),
            "conf_range": list(self.conf_range),
        }
        if self.session_dir is not None:
            summary["session_dir"] = str(self.session_dir)
        if self.sensor_valid_ratio is not None:
            summary["sensor_nonzero_ratio"] = self.sensor_valid_ratio
        if self.sensor_depth_range is not None:
            summary["sensor_depth_range"] = list(self.sensor_depth_range)
        if self.fusion is not None:
            summary["fusion"] = self.fusion
            summary["fusion_sensor_valid_ratio"] = self.fusion.get("sensor_valid_ratio")
        if self.point_count is not None:
            summary["point_count"] = self.point_count
        if self.sensor_point_count is not None:
            summary["sensor_point_count"] = self.sensor_point_count
        if self.rgb_shift is not None:
            summary["rgb_shift"] = self.rgb_shift
        return summary


class DepthService:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        device: str | None = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        session_dir: Path = DEFAULT_SESSION_DIR,
        max_points: int = DEFAULT_MAX_POINTS,
        save_uploads: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.session_dir = Path(session_dir)
        self.max_points = max_points
        self.save_uploads = save_uploads
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
        *,
        source: str = "unknown",
        depth_path: Path | str | None = None,
        depth_bytes: bytes | None = None,
        intrinsics_path: Path | str | None = None,
        intrinsics_data: dict | None = None,
        rgb_bytes: bytes | None = None,
        rgb_suffix: str = ".png",
        save_session: bool | None = None,
        rgb_shift_x: float | int | None = None,
        rgb_shift_y: float | int | None = None,
    ) -> DepthResult:
        rgb_input = str(rgb) if isinstance(rgb, (str, Path)) else rgb
        rgb_image = Image.open(rgb_input).convert("RGB") if isinstance(rgb_input, str) else rgb_input.convert("RGB")
        rgb_array = np.array(rgb_image)

        sx, sy, shift_src = resolve_rgb_shift_xy(
            rgb_shift_x,
            rgb_shift_y,
            intrinsics_data if isinstance(intrinsics_data, dict) else None,
        )
        rgb_shift_meta = {"dx": sx, "dy": sy, "source": shift_src}
        rgb_shift_xy = (sx, sy)

        intrinsics_batch = intrinsics[None] if intrinsics is not None else None
        prediction = self.model.inference(
            [rgb_input if isinstance(rgb_input, str) else rgb_image],
            intrinsics=intrinsics_batch,
        )

        pred_depth = prediction.depth[0]
        pred_conf = prediction.conf[0]

        sensor_depth_vis = None
        sensor_valid_ratio = None
        sensor_depth_range = None
        if sensor_depth is not None:
            sensor_depth_vis = depth_to_colormap_with_colorbar(sensor_depth, unit="mm")
            valid = np.isfinite(sensor_depth) & (sensor_depth > 0)
            if valid.any():
                sensor_valid_ratio = float(valid.mean())
                sensor_depth_range = (
                    float(np.nanmin(sensor_depth)),
                    float(np.nanmax(sensor_depth)),
                )

        fusion_result: FusionResult | None = None
        fused_depth_vis = None
        fusion_mask_vis = None
        if sensor_depth is not None:
            fusion_result = fuse_sensor_and_estimated(
                sensor_depth=sensor_depth,
                est_depth=pred_depth,
                rgb=rgb_array,
                conf=pred_conf,
            )
            fused_depth_vis = depth_to_colormap_with_colorbar(fusion_result.fused_depth, unit="mm")
            fusion_mask_vis = fusion_region_to_vis(
                fusion_result.valid_mask,
                fusion_result.inpaint_mask,
            )

        pointcloud_glb = None
        pointcloud_ply = None
        point_count = None
        sensor_pointcloud_glb = None
        sensor_pointcloud_ply = None
        sensor_point_count = None
        pc_intrinsics = intrinsics
        pc_depth = pred_depth
        pc_rgb = prediction.processed_images[0]
        pc_conf = pred_conf
        pc_stem = None

        if fusion_result is not None and intrinsics is not None:
            pc_depth = fusion_result.fused_depth
            pc_rgb = rgb_array
            pc_conf = None
        elif intrinsics is not None and prediction.intrinsics is not None:
            pc_intrinsics = prediction.intrinsics[0].astype(np.float32)
        elif intrinsics is not None:
            if isinstance(rgb_input, str):
                orig_w, orig_h = Image.open(rgb_input).size
            elif isinstance(rgb_input, Image.Image):
                orig_w, orig_h = rgb_input.size
            else:
                orig_h, orig_w = pred_depth.shape
            from src.depth.pointcloud import scale_intrinsics

            pc_intrinsics = scale_intrinsics(
                intrinsics,
                src_size=(orig_w, orig_h),
                dst_size=(pred_depth.shape[1], pred_depth.shape[0]),
            )

        if pc_intrinsics is not None:
            pc_stem = uuid.uuid4().hex[:12]
            pc_info = build_pointcloud_files(
                depth=pc_depth,
                rgb=pc_rgb,
                intrinsics=pc_intrinsics,
                output_dir=self.output_dir,
                max_points=self.max_points,
                conf=pc_conf,
                stem=f"{pc_stem}_fused" if fusion_result is not None else pc_stem,
                rgb_shift_xy=rgb_shift_xy,
            )
            pointcloud_glb = pc_info["glb_path"]
            pointcloud_ply = pc_info["ply_path"]
            point_count = pc_info["point_count"]

            if sensor_depth is not None and fusion_result is not None:
                sensor_pc_info = build_pointcloud_files(
                    depth=sensor_depth,
                    rgb=rgb_array,
                    intrinsics=intrinsics,
                    output_dir=self.output_dir,
                    max_points=self.max_points,
                    conf=None,
                    stem=f"{pc_stem}_sensor",
                    rgb_shift_xy=rgb_shift_xy,
                )
                sensor_pointcloud_glb = sensor_pc_info["glb_path"]
                sensor_pointcloud_ply = sensor_pc_info["ply_path"]
                sensor_point_count = sensor_pc_info["point_count"]

        result = DepthResult(
            pred_depth=pred_depth,
            pred_conf=pred_conf,
            processed_rgb=Image.fromarray(prediction.processed_images[0]),
            pred_depth_vis=depth_to_colormap_with_colorbar(pred_depth, unit="相对"),
            conf_vis=conf_to_colormap(pred_conf),
            depth_range=(float(pred_depth.min()), float(pred_depth.max())),
            conf_range=(float(pred_conf.min()), float(pred_conf.max())),
            intrinsics=intrinsics,
            sensor_depth_vis=sensor_depth_vis,
            sensor_valid_ratio=sensor_valid_ratio,
            sensor_depth_range=sensor_depth_range,
            fused_depth=fusion_result.fused_depth if fusion_result else None,
            fused_depth_vis=fused_depth_vis,
            fusion_mask_vis=fusion_mask_vis,
            fusion=fusion_result.to_summary() if fusion_result else None,
            pointcloud_glb=pointcloud_glb,
            pointcloud_ply=pointcloud_ply,
            point_count=point_count,
            sensor_pointcloud_glb=sensor_pointcloud_glb,
            sensor_pointcloud_ply=sensor_pointcloud_ply,
            sensor_point_count=sensor_point_count,
            rgb_shift=rgb_shift_meta,
        )

        should_archive = self.save_uploads if save_session is None else save_session
        if should_archive:
            archived_dir = archive_session(
                rgb=rgb_image,
                result=result,
                source=source,
                base_dir=self.session_dir,
                depth_path=depth_path,
                depth_bytes=depth_bytes,
                depth_array=sensor_depth if depth_path is None and depth_bytes is None else None,
                intrinsics_path=intrinsics_path,
                intrinsics_data=intrinsics_data,
                rgb_bytes=rgb_bytes,
                rgb_suffix=rgb_suffix,
            )
            result.session_dir = archived_dir

        return result
