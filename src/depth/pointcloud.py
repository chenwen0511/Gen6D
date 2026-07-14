"""Point cloud generation and export utilities."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from PIL import Image


DEFAULT_MAX_POINTS = 50_000
CAMERA_TO_GLB = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
_POSE_AXIS_COLORS = (
    [255, 64, 64, 255],
    [64, 220, 64, 255],
    [64, 96, 255, 255],
)
# 点云角色：0=背景, 1=分割 mask 内, 2=PEM 球裁剪内（参与位姿）
_ROLE_BG = 0
_ROLE_SEG = 1
_ROLE_PEM = 2
_ROLE_COLORS = {
    _ROLE_BG: np.array([88, 88, 96], dtype=np.uint8),
    _ROLE_SEG: np.array([255, 140, 30], dtype=np.uint8),
    _ROLE_PEM: np.array([40, 255, 100], dtype=np.uint8),
}


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


def _subsample_by_role(
    points: np.ndarray,
    colors: np.ndarray,
    roles: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = points.shape[0]
    if n <= max_points:
        return points, colors, roles

    rng = np.random.default_rng(42)
    keep = np.zeros(n, dtype=bool)
    remaining = max_points

    for role in (_ROLE_PEM, _ROLE_SEG, _ROLE_BG):
        idx = np.flatnonzero(roles == role)
        if idx.size == 0:
            continue
        if idx.size <= remaining:
            keep[idx] = True
            remaining -= int(idx.size)
        else:
            chosen = rng.choice(idx, remaining, replace=False)
            keep[chosen] = True
            remaining = 0
            break

    return points[keep], colors[keep], roles[keep]


def depth_rgb_to_role_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    *,
    segment_mask: np.ndarray | None = None,
    pose_cam: tuple[np.ndarray, np.ndarray] | None = None,
    pem_sphere_radius_mm: float | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    生成带角色标签的点云：背景 / 分割 mask / PEM 球裁剪（近似 SAM-6D 参与点）。

    ``pose_cam`` 为相机系 (t_mm, R_cam_obj)，满足 ``p_cam = R @ p_obj + t``。
    """
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb_img = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
        rgb = np.array(rgb_img)

    if segment_mask is not None and segment_mask.shape != (h, w):
        segment_mask = np.array(
            Image.fromarray(segment_mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        ) > 127

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), {
            "background": 0,
            "segmentation": 0,
            "pem_crop": 0,
        }

    z = depth[valid].astype(np.float32)
    x_cam = (u_coords[valid] - cx) * z / fx
    y_cam = (v_coords[valid] - cy) * z / fy
    points_cam = np.stack([x_cam, y_cam, z], axis=-1)

    roles = np.zeros(points_cam.shape[0], dtype=np.uint8)
    if segment_mask is not None:
        roles[segment_mask[valid]] = _ROLE_SEG

    if (
        segment_mask is not None
        and pose_cam is not None
        and pem_sphere_radius_mm is not None
        and pem_sphere_radius_mm > 0
    ):
        t_mm, rotation_cam = pose_cam
        t = np.asarray(t_mm, dtype=np.float64)
        r = np.asarray(rotation_cam, dtype=np.float64)
        seg_idx = roles == _ROLE_SEG
        if seg_idx.any():
            p_obj = (points_cam[seg_idx] - t) @ r
            in_sphere = np.linalg.norm(p_obj, axis=1) <= float(pem_sphere_radius_mm)
            seg_indices = np.flatnonzero(seg_idx)
            roles[seg_indices[in_sphere]] = _ROLE_PEM

    colors = np.zeros((points_cam.shape[0], 3), dtype=np.uint8)
    for role, color in _ROLE_COLORS.items():
        colors[roles == role] = color

    points_glb = points_cam.copy()
    points_glb[:, 1] *= -1

    points_glb, colors, roles = _subsample_by_role(points_glb, colors, roles, max_points)

    stats = {
        "background": int(np.count_nonzero(roles == _ROLE_BG)),
        "segmentation": int(np.count_nonzero(roles == _ROLE_SEG)),
        "pem_crop": int(np.count_nonzero(roles == _ROLE_PEM)),
    }
    return points_glb, colors, stats


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


def euler_zyx_to_rotation_matrix(euler_zyx_rad: Sequence[float]) -> np.ndarray:
    z, y, x = (float(v) for v in euler_zyx_rad)
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return rz @ ry @ rx


def camera_pose_mm_to_glb(
    position_mm: Sequence[float],
    rotation_3x3: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """相机系（Y 向下）→ GLB/Three.js（Y 向上）。"""
    t = CAMERA_TO_GLB @ np.asarray(position_mm, dtype=np.float64)
    r = CAMERA_TO_GLB @ np.asarray(rotation_3x3, dtype=np.float64) @ CAMERA_TO_GLB
    return t, r


def _align_z_to_direction(direction: np.ndarray) -> np.ndarray:
    direction = direction.astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.eye(4)
    direction = direction / norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.allclose(direction, z_axis):
        return np.eye(4)
    if np.allclose(direction, -z_axis):
        return trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
    return trimesh.transformations.rotation_matrix(angle, axis)


def create_pose_axes_mesh(
    origin: np.ndarray,
    rotation_glb: np.ndarray,
    *,
    axis_length_mm: float = 80.0,
    radius_mm: float = 2.0,
) -> trimesh.Trimesh:
    parts: list[trimesh.Trimesh] = []
    origin = np.asarray(origin, dtype=np.float64)
    for axis_idx, color in enumerate(_POSE_AXIS_COLORS):
        direction = rotation_glb[:, axis_idx]
        cyl = trimesh.creation.cylinder(radius=radius_mm, height=axis_length_mm, sections=10)
        cyl.apply_transform(_align_z_to_direction(direction))
        cyl.apply_translation(origin + direction * (axis_length_mm / 2.0))
        cyl.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(cyl.faces), 1))
        parts.append(cyl)

    marker = trimesh.creation.icosphere(radius=radius_mm * 2.5, subdivisions=2)
    marker.apply_translation(origin)
    marker.visual.face_colors = np.tile(
        np.array([255, 220, 0, 255], dtype=np.uint8),
        (len(marker.faces), 1),
    )
    parts.append(marker)
    return trimesh.util.concatenate(parts)


def export_scene_glb(geometries: Sequence[trimesh.Trimesh | trimesh.points.PointCloud], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for geom in geometries:
        scene.add_geometry(geom)
    scene.export(output_path)
    return output_path


def build_pointcloud_scene_files(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    output_dir: Path,
    *,
    poses: Sequence[tuple[Sequence[float], Sequence[Sequence[float]]]] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    conf: np.ndarray | None = None,
    segment_mask: np.ndarray | None = None,
    pem_sphere_radius_mm: float | None = None,
    stem: str | None = None,
) -> dict:
    """生成融合点云 GLB，并可选叠加 PEM 位姿坐标轴与分割/PEM 参与点着色。"""
    pose_cam = poses[0] if poses else None
    if segment_mask is not None:
        points, colors, role_stats = depth_rgb_to_role_pointcloud(
            depth=depth,
            rgb=rgb,
            intrinsics=intrinsics,
            segment_mask=segment_mask,
            pose_cam=pose_cam,
            pem_sphere_radius_mm=pem_sphere_radius_mm,
            max_points=max_points,
        )
    else:
        points, colors = depth_rgb_to_pointcloud(
            depth=depth,
            rgb=rgb,
            intrinsics=intrinsics,
            max_points=max_points,
            conf=conf,
        )
        role_stats = {"background": int(points.shape[0]), "segmentation": 0, "pem_crop": 0}

    stem = stem or uuid.uuid4().hex[:12]
    glb_path = output_dir / f"{stem}_scene.glb"
    ply_path = output_dir / f"{stem}.ply"
    export_pointcloud_ply(points, colors, ply_path)

    if points.size == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=[[128, 128, 128]])
    else:
        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)

    geometries: list[trimesh.Trimesh | trimesh.points.PointCloud] = [cloud]
    if poses:
        for position_mm, rotation_cam in poses:
            origin, rotation_glb = camera_pose_mm_to_glb(position_mm, rotation_cam)
            geometries.append(
                create_pose_axes_mesh(origin, rotation_glb, axis_length_mm=80.0, radius_mm=2.0)
            )

    export_scene_glb(geometries, glb_path)
    return {
        "point_count": int(points.shape[0]),
        "glb_path": glb_path,
        "ply_path": ply_path,
        "pose_count": len(poses or ()),
        "point_roles": role_stats,
        "segmentation_applied": segment_mask is not None,
        "pem_sphere_radius_mm": pem_sphere_radius_mm,
    }
