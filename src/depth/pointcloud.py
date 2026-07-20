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


def create_pose_marker_sphere(
    origin: np.ndarray,
    *,
    radius_mm: float = 10.0,
    color_rgba: Sequence[int] = (255, 0, 255, 255),
) -> trimesh.Trimesh:
    """醒目球体，用于标出 p_i / 关键点。"""
    origin = np.asarray(origin, dtype=np.float64)
    sphere = trimesh.creation.icosphere(radius=float(radius_mm), subdivisions=2)
    sphere.apply_translation(origin)
    rgba = np.array(color_rgba, dtype=np.uint8)
    if rgba.shape[0] == 3:
        rgba = np.concatenate([rgba, np.array([255], dtype=np.uint8)])
    sphere.visual.face_colors = np.tile(rgba, (len(sphere.faces), 1))
    return sphere


def inject_axis_points(
    origin_glb: np.ndarray,
    rotation_glb: np.ndarray,
    *,
    axis_length_mm: float = 50.0,
    samples_per_axis: int = 24,
    core_color: Sequence[int] = (255, 0, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """
    用点列画出原点 + 三轴（兼容 Gradio 只渲染 PointCloud 的情况）。
    返回 (M,3) 点、(M,3) RGB。
    """
    origin_glb = np.asarray(origin_glb, dtype=np.float64)
    rotation_glb = np.asarray(rotation_glb, dtype=np.float64)
    axis_colors = (
        np.array([255, 64, 64], dtype=np.uint8),
        np.array([64, 220, 64], dtype=np.uint8),
        np.array([64, 96, 255], dtype=np.uint8),
    )
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    # 原点密集球状点云
    rng = np.random.default_rng(0)
    core = origin_glb + rng.normal(0.0, 2.5, size=(80, 3))
    pts.append(core)
    cols.append(np.tile(np.asarray(core_color, dtype=np.uint8), (core.shape[0], 1)))
    for axis_idx, color in enumerate(axis_colors):
        direction = rotation_glb[:, axis_idx]
        for t in np.linspace(0.0, axis_length_mm, samples_per_axis):
            pts.append(origin_glb + direction * float(t))
            cols.append(color)
    return np.vstack(pts).astype(np.float32), np.vstack(cols).astype(np.uint8)


def export_scene_glb(geometries: Sequence[trimesh.Trimesh | trimesh.points.PointCloud], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for geom in geometries:
        scene.add_geometry(geom)
    scene.export(output_path)
    return output_path


def depth_instance_to_pointcloud(
    depth: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    instance_colors_rgb: Sequence[Sequence[int]],
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    background_color: Sequence[int] = (88, 88, 96),
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    传感器深度反投影，按实例 id mask 分色。

    ``instance_id_map``：H×W，0=背景，1..N=实例；``instance_colors_rgb`` 按 id-1 取色。
    """
    h, w = depth.shape
    if instance_id_map.shape != (h, w):
        id_img = Image.fromarray(instance_id_map.astype(np.uint8)).resize((w, h), Image.NEAREST)
        instance_id_map = np.array(id_img, dtype=np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"background": 0, "instances": 0},
        )

    z = depth[valid].astype(np.float32)
    x = (u_coords[valid] - cx) * z / fx
    y = (v_coords[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    points[:, 1] *= -1

    ids = instance_id_map[valid].astype(np.int32)
    colors = np.tile(np.asarray(background_color, dtype=np.uint8), (points.shape[0], 1))
    palette = [np.asarray(c, dtype=np.uint8)[:3] for c in instance_colors_rgb]
    for inst_id in np.unique(ids):
        if int(inst_id) <= 0:
            continue
        color = palette[(int(inst_id) - 1) % max(len(palette), 1)] if palette else np.array([255, 140, 30], dtype=np.uint8)
        colors[ids == inst_id] = color

    # 采样时优先保留实例点
    n = points.shape[0]
    if n > max_points:
        rng = np.random.default_rng(42)
        keep = np.zeros(n, dtype=bool)
        remaining = max_points
        for mask in (ids > 0, ids == 0):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            if idx.size <= remaining:
                keep[idx] = True
                remaining -= int(idx.size)
            else:
                keep[rng.choice(idx, remaining, replace=False)] = True
                remaining = 0
                break
        points = points[keep]
        colors = colors[keep]
        ids = ids[keep]

    return points, colors, {
        "background": int(np.count_nonzero(ids == 0)),
        "instances": int(np.count_nonzero(ids > 0)),
        "instance_ids": int(len(np.unique(ids[ids > 0]))),
    }


def depth_instance_points_camera(
    depth: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    反投影到相机系（Y 向下，Z 向前，单位 mm），返回 (N,3) 点与 (N,) 实例 id。
    不做 Y 翻转、不降采样，供实例内离群剔除 / min-Z 分析。
    """
    h, w = depth.shape
    if instance_id_map.shape != (h, w):
        id_img = Image.fromarray(instance_id_map.astype(np.uint8)).resize((w, h), Image.NEAREST)
        instance_id_map = np.array(id_img, dtype=np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = np.isfinite(depth) & (depth > 0) & (instance_id_map > 0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int32)

    z = depth[valid].astype(np.float64)
    x = (u_coords[valid] - cx) * z / fx
    y = (v_coords[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    ids = instance_id_map[valid].astype(np.int32)
    return points, ids


def remove_statistical_outliers(
    points: np.ndarray,
    *,
    std_ratio: float = 2.0,
    z_mad_ratio: float = 2.5,
) -> np.ndarray:
    """对单实例点云做简单统计离群剔除（质心距离 + Z 的 MAD）。"""
    if points.ndim != 2 or points.shape[0] < 8:
        return points

    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    d_med = float(np.median(dist))
    d_mad = float(np.median(np.abs(dist - d_med))) + 1e-6
    keep = dist <= (d_med + std_ratio * 1.4826 * d_mad)

    z = points[:, 2]
    z_med = float(np.median(z[keep])) if keep.any() else float(np.median(z))
    z_mad = float(np.median(np.abs(z[keep] - z_med))) + 1e-6 if keep.any() else 1e-6
    keep &= np.abs(z - z_med) <= (z_mad_ratio * 1.4826 * z_mad)

    if int(keep.sum()) < 3:
        return points
    return points[keep]


def find_instance_min_z_points(
    depth: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    std_ratio: float = 2.0,
    z_mad_ratio: float = 2.5,
) -> list[dict]:
    """
    对每个实例：剔除离群点 → 取相机系 Z（深度）最小的点 p_i（最近）。

    :return: [{instance_id, position_mm, num_raw, num_inlier, z_mm}, ...]
    """
    points, ids = depth_instance_points_camera(depth, instance_id_map, intrinsics)
    results: list[dict] = []
    if points.size == 0:
        return results

    for inst_id in sorted(int(v) for v in np.unique(ids) if int(v) > 0):
        pts = points[ids == inst_id]
        raw_n = int(pts.shape[0])
        inliers = remove_statistical_outliers(pts, std_ratio=std_ratio, z_mad_ratio=z_mad_ratio)
        if inliers.shape[0] == 0:
            continue
        idx = int(np.argmin(inliers[:, 2]))
        p_i = inliers[idx]
        results.append(
            {
                "instance_id": inst_id,
                "position_mm": [round(float(p_i[0]), 2), round(float(p_i[1]), 2), round(float(p_i[2]), 2)],
                "z_mm": round(float(p_i[2]), 2),
                "num_raw": raw_n,
                "num_inlier": int(inliers.shape[0]),
            }
        )
    return results


def _qi_xy_mean_z_from_pi(filter_pts: np.ndarray, p_i: np.ndarray) -> np.ndarray:
    """筛选点取 xy 均值，z 固定为 p_i（instance min-Z）。"""
    if filter_pts.shape[0] == 0:
        return p_i.copy()
    xy_mean = filter_pts[:, :2].mean(axis=0)
    return np.array([xy_mean[0], xy_mean[1], float(p_i[2])], dtype=np.float64)


def find_instance_qi_from_pi_sphere(
    depth: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    radius_mm: float = 8.0,
    y_band_mm: float = 2.0,
    std_ratio: float = 2.0,
    z_mad_ratio: float = 2.5,
) -> list[dict]:
    """
    各实例：先按原规则取 p_i（剔除外点后 Z 最小），再以 p_i 为球心、
    半径 radius_mm 内的点，再取相机系 |y - p_i.y| ≤ y_band_mm 的点；
    q_i 的 xy 为筛选点均值，z 取 p_i.z（min-Z）。若 y 带内无点则回退球内 xy 均值。
    ``position_mm`` 为 q_i（供 UI 展示）。

    :return: [{instance_id, position_mm(=q_i), p_i_mm, q_i_mm, radius_mm, y_band_mm,
               num_sphere, num_band, num_raw, num_inlier, z_mm}, ...]
    """
    points, ids = depth_instance_points_camera(depth, instance_id_map, intrinsics)
    results: list[dict] = []
    if points.size == 0:
        return results

    radius = float(radius_mm)
    band = float(y_band_mm)
    for inst_id in sorted(int(v) for v in np.unique(ids) if int(v) > 0):
        pts = points[ids == inst_id]
        raw_n = int(pts.shape[0])
        inliers = remove_statistical_outliers(pts, std_ratio=std_ratio, z_mad_ratio=z_mad_ratio)
        if inliers.shape[0] == 0:
            continue
        idx = int(np.argmin(inliers[:, 2]))
        p_i = inliers[idx]
        y_ref = float(p_i[1])
        dist = np.linalg.norm(inliers - p_i, axis=1)
        sphere_mask = dist <= radius
        sphere_pts = inliers[sphere_mask]
        num_sphere = int(sphere_pts.shape[0])
        if num_sphere == 0:
            q_i = p_i
            num_band = 0
        else:
            band_mask = np.abs(sphere_pts[:, 1] - y_ref) <= band
            band_pts = sphere_pts[band_mask]
            num_band = int(band_pts.shape[0])
            if num_band == 0:
                q_i = _qi_xy_mean_z_from_pi(sphere_pts, p_i)
            else:
                q_i = _qi_xy_mean_z_from_pi(band_pts, p_i)
        results.append(
            {
                "instance_id": inst_id,
                "position_mm": [
                    round(float(q_i[0]), 2),
                    round(float(q_i[1]), 2),
                    round(float(q_i[2]), 2),
                ],
                "p_i_mm": [
                    round(float(p_i[0]), 2),
                    round(float(p_i[1]), 2),
                    round(float(p_i[2]), 2),
                ],
                "q_i_mm": [
                    round(float(q_i[0]), 2),
                    round(float(q_i[1]), 2),
                    round(float(q_i[2]), 2),
                ],
                "radius_mm": radius,
                "y_band_mm": band,
                "y_ref_mm": round(y_ref, 2),
                "num_sphere": num_sphere,
                "num_band": num_band,
                "z_mm": round(float(q_i[2]), 2),
                "num_raw": raw_n,
                "num_inlier": int(inliers.shape[0]),
            }
        )
    return results


def find_instance_qi_from_pi_y_band(
    depth: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    y_band_mm: float = 2.0,
    radius_mm: float = 8.0,
    std_ratio: float = 2.0,
    z_mad_ratio: float = 2.5,
) -> list[dict]:
    """兼容旧名：球半径 radius_mm + 相机 y ± y_band_mm 两步聚合 q_i。"""
    return find_instance_qi_from_pi_sphere(
        depth,
        instance_id_map,
        intrinsics,
        radius_mm=float(radius_mm),
        y_band_mm=float(y_band_mm),
        std_ratio=std_ratio,
        z_mad_ratio=z_mad_ratio,
    )


# 兼容旧名
find_instance_max_z_points = find_instance_min_z_points


def find_instance_nearest_p1_x_points(
    depth: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    p1_position_mm: np.ndarray | Sequence[float],
    p1_rotation: np.ndarray | Sequence[Sequence[float]],
    *,
    std_ratio: float = 2.0,
    z_mad_ratio: float = 2.5,
) -> list[dict]:
    """
    对每个实例：剔除外点后，取沿 P1 局部 X 轴距离 |((p-p1)·X)| 最小的点。

    :return: [{instance_id, position_mm, x_dist_mm, z_mm, num_raw, num_inlier}, ...]
    """
    p1 = np.asarray(p1_position_mm, dtype=np.float64).reshape(3)
    rot = np.asarray(p1_rotation, dtype=np.float64).reshape(3, 3)
    x_axis = rot[:, 0]
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-12:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm

    points, ids = depth_instance_points_camera(depth, instance_id_map, intrinsics)
    results: list[dict] = []
    if points.size == 0:
        return results

    for inst_id in sorted(int(v) for v in np.unique(ids) if int(v) > 0):
        pts = points[ids == inst_id]
        raw_n = int(pts.shape[0])
        inliers = remove_statistical_outliers(pts, std_ratio=std_ratio, z_mad_ratio=z_mad_ratio)
        if inliers.shape[0] == 0:
            continue
        x_dist = np.abs((inliers - p1) @ x_axis)
        idx = int(np.argmin(x_dist))
        p_sel = inliers[idx]
        results.append(
            {
                "instance_id": inst_id,
                "position_mm": [
                    round(float(p_sel[0]), 2),
                    round(float(p_sel[1]), 2),
                    round(float(p_sel[2]), 2),
                ],
                "x_dist_mm": round(float(x_dist[idx]), 2),
                "z_mm": round(float(p_sel[2]), 2),
                "num_raw": raw_n,
                "num_inlier": int(inliers.shape[0]),
            }
        )
    return results


def annotate_p1_x_distance(
    items: list[dict],
    p1_position_mm: np.ndarray | Sequence[float],
    p1_rotation: np.ndarray | Sequence[Sequence[float]],
) -> list[dict]:
    """为每个点写入沿 P1 局部 X 的 |((p-p1)·X)|（mm）。"""
    p1 = np.asarray(p1_position_mm, dtype=np.float64).reshape(3)
    rot = np.asarray(p1_rotation, dtype=np.float64).reshape(3, 3)
    x_axis = rot[:, 0]
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-12:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis = x_axis / x_norm

    out: list[dict] = []
    for item in items:
        row = dict(item)
        pos = np.asarray(row.get("position_mm"), dtype=np.float64).reshape(3)
        row["x_dist_mm"] = round(float(abs((pos - p1) @ x_axis)), 2)
        out.append(row)
    return out


def select_nearest_along_p1_x(items: list[dict]) -> list[dict]:
    """
    按 |dx| = x_dist_mm（沿 P1-X）排序，只保留最近的 1 个；其余丢弃。
    需要条目上已有 x_dist_mm。
    """
    if not items:
        return []
    ranked = sorted(items, key=lambda r: float(r.get("x_dist_mm", float("inf"))))
    best = dict(ranked[0])
    best["selected_by"] = "nearest_p1_x"
    best["rank"] = 1
    best["candidates"] = len(ranked)
    return [best]


def build_instance_pointcloud_files(
    depth: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    instance_colors_rgb: Sequence[Sequence[int]],
    output_dir: Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    stem: str | None = None,
) -> dict:
    """生成按 SAM3 实例分色的传感器深度点云 GLB/PLY。"""
    points, colors, stats = depth_instance_to_pointcloud(
        depth=depth,
        instance_id_map=instance_id_map,
        intrinsics=intrinsics,
        instance_colors_rgb=instance_colors_rgb,
        max_points=max_points,
    )
    stem = stem or uuid.uuid4().hex[:12]
    glb_path = output_dir / f"{stem}_instance.glb"
    ply_path = output_dir / f"{stem}_instance.ply"
    export_pointcloud_ply(points, colors, ply_path)
    export_pointcloud_glb(points, colors, glb_path)
    return {
        "point_count": int(points.shape[0]),
        "glb_path": glb_path,
        "ply_path": ply_path,
        "point_stats": stats,
    }


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
