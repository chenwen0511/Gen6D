"""Point cloud generation and export utilities."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


DEFAULT_MAX_POINTS = 50_000


def scale_intrinsics(
    intrinsics: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = dst_w / src_w
    sy = dst_h / src_h
    k = intrinsics.copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def depth_rgb_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    max_points: int = DEFAULT_MAX_POINTS,
    conf: np.ndarray | None = None,
    conf_percentile: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb_img = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
        rgb = np.array(rgb_img)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        thresh = np.percentile(conf[valid], conf_percentile) if valid.any() else 0
        valid &= conf >= thresh

    if not valid.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth[valid].astype(np.float32)
    x = (u_coords[valid] - cx) * z / fx
    y = (v_coords[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    # Camera coords: Y down, Z forward → GLB/Three.js: Y up
    points[:, 1] *= -1

    colors = rgb[valid]
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    if colors.shape[-1] == 4:
        colors = colors[..., :3]

    n_points = points.shape[0]
    if n_points > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_points, max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    return points, colors


def export_pointcloud_glb(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if points.size == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=[[128, 128, 128]])
    else:
        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)

    cloud.export(output_path)
    return output_path


def export_pointcloud_ply(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if points.size == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=[[128, 128, 128]])
    else:
        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)

    cloud.export(output_path)
    return output_path


def build_pointcloud_files(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: Path,
    max_points: int = DEFAULT_MAX_POINTS,
    conf: np.ndarray | None = None,
    stem: str | None = None,
) -> dict:
    points, colors = depth_rgb_to_pointcloud(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        max_points=max_points,
        conf=conf,
    )

    stem = stem or uuid.uuid4().hex[:12]
    glb_path = output_dir / f"{stem}.glb"
    ply_path = output_dir / f"{stem}.ply"
    export_pointcloud_glb(points, colors, glb_path)
    export_pointcloud_ply(points, colors, ply_path)

    return {
        "point_count": int(points.shape[0]),
        "glb_path": glb_path,
        "ply_path": ply_path,
    }
