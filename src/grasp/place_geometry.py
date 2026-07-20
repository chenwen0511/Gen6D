"""放置阶段几何：深度反投影、位姿估计与偏移。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

Array3 = np.ndarray

# 相机坐标系：X 右、Y 下、Z 前（视线深度方向）
CAMERA_AXIS_X = np.array([1.0, 0.0, 0.0], dtype=np.float64)
CAMERA_AXIS_Y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
CAMERA_AXIS_Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
CAMERA_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # 图像向上 = -Y


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class Pose6D:
    """相机坐标系下的 6D 位姿。"""

    position_m: np.ndarray
    rotation: np.ndarray

    @property
    def position_mm(self) -> np.ndarray:
        return self.position_m * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_m": self.position_m.round(6).tolist(),
            "position_mm": self.position_mm.round(3).tolist(),
            "rotation_matrix": self.rotation.round(6).tolist(),
            "rotation_euler_zyx_rad": rotation_matrix_to_euler_zyx(self.rotation),
        }


def load_camera_json(path: Path) -> CameraIntrinsics:
    data = json.loads(path.read_text(encoding="utf-8"))
    cam_k = data.get("cam_K")
    if not isinstance(cam_k, list) or len(cam_k) != 9:
        raise ValueError(f"camera.json cam_K 必须为 9 元素数组: {path}")
    depth_scale = float(data.get("depth_scale", 0.001))
    return CameraIntrinsics(
        fx=float(cam_k[0]),
        fy=float(cam_k[4]),
        cx=float(cam_k[2]),
        cy=float(cam_k[5]),
        depth_scale=depth_scale,
    )


def load_depth_meters(path: Path, *, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"无法读取 depth: {path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]

    depth_f = depth_raw.astype(np.float64)
    depth_m = depth_f * float(depth_scale)

    valid = depth_f > 0
    if valid.any():
        median_z = float(np.median(depth_m[valid]))
        if float(depth_scale) >= 0.5 and median_z > 5.0:
            depth_m = depth_f / 1000.0
    return depth_m


def depth_to_meters(raw: float, depth_scale: float) -> float:
    if raw <= 0:
        return float("nan")
    z = float(raw) * float(depth_scale)
    if float(depth_scale) >= 0.5 and z > 5.0:
        return float(raw) / 1000.0
    return z


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("mask 为空，无法计算质心")
    return float(xs.mean()), float(ys.mean())


def _valid_depths_in_neighborhood(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int,
) -> List[float]:
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    values: List[float] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = ui + dx, vi + dy
            if 0 <= x < w and 0 <= y < h:
                z = float(depth_m[y, x])
                if z > 0 and np.isfinite(z):
                    values.append(z)
    return values


def _sample_depth_from_mask_nearby(
    mask: np.ndarray,
    depth_m: np.ndarray,
    u: float,
    v: float,
    *,
    k: int = 5,
) -> float:
    """在 mask 内取距 (u,v) 最近的有效深度像素（中位数）。"""
    ys, xs = np.where(mask)
    zs = depth_m[ys, xs].astype(np.float64)
    valid = (zs > 0) & np.isfinite(zs)
    if not valid.any():
        raise ValueError(f"({u:.1f}, {v:.1f}) 附近无有效深度")
    xs_v = xs[valid].astype(np.float64)
    ys_v = ys[valid].astype(np.float64)
    zs_v = zs[valid]
    dist2 = (xs_v - u) ** 2 + (ys_v - v) ** 2
    order = np.argsort(dist2)
    take = order[: min(k, len(order))]
    return float(np.median(zs_v[take]))


def sample_depth_at(
    depth_m: np.ndarray,
    u: float,
    v: float,
    *,
    radius: int = 2,
    max_radius: int = 32,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    在 (u,v) 采样深度：先查邻域，再逐步扩大搜索半径，最后回退到 mask 内最近有效像素。
    """
    for r in range(radius, max_radius + 1):
        values = _valid_depths_in_neighborhood(depth_m, u, v, r)
        if values:
            return float(np.median(values))
    if mask is not None:
        return _sample_depth_from_mask_nearby(mask, depth_m, u, v)
    raise ValueError(f"({u:.1f}, {v:.1f}) 附近无有效深度")


def marker_depth_diagnostics(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    neighborhood_radius: int = 2,
) -> Dict[str, Any]:
    """统计 marker mask 区域与质心邻域的深度有效性，便于 UI 报错定位。"""
    u, v = mask_centroid(mask)
    ys, xs = np.where(mask)
    mask_depths = depth_m[ys, xs].astype(np.float64)
    mask_valid = (mask_depths > 0) & np.isfinite(mask_depths)

    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    nb_values: List[float] = []
    for dy in range(-neighborhood_radius, neighborhood_radius + 1):
        for dx in range(-neighborhood_radius, neighborhood_radius + 1):
            x, y = ui + dx, vi + dy
            if 0 <= x < w and 0 <= y < h:
                z = float(depth_m[y, x])
                nb_values.append(z)

    nb_arr = np.asarray(nb_values, dtype=np.float64)
    nb_valid = (nb_arr > 0) & np.isfinite(nb_arr)

    diag: Dict[str, Any] = {
        "centroid_uv": [round(u, 2), round(v, 2)],
        "neighborhood_radius_px": neighborhood_radius,
        "mask_pixels": int(len(xs)),
        "mask_valid_depth_pixels": int(mask_valid.sum()),
        "mask_valid_depth_ratio": round(float(mask_valid.mean()), 4) if mask_valid.size else 0.0,
        "centroid_neighborhood_total_pixels": int(nb_arr.size),
        "centroid_neighborhood_valid_pixels": int(nb_valid.sum()),
    }
    if mask_valid.any():
        diag["mask_depth_m_median"] = round(float(np.median(mask_depths[mask_valid])), 4)
    if nb_valid.any():
        diag["centroid_neighborhood_depth_m_median"] = round(float(np.median(nb_arr[nb_valid])), 4)
    return diag


def backproject(u: float, v: float, z_m: float, camera: CameraIntrinsics) -> np.ndarray:
    x = (u - camera.cx) * z_m / camera.fx
    y = (v - camera.cy) * z_m / camera.fy
    return np.array([x, y, z_m], dtype=np.float64)


def project_point(point_m: np.ndarray, camera: CameraIntrinsics) -> Optional[Tuple[int, int]]:
    z = float(point_m[2])
    if z <= 1e-6:
        return None
    u = camera.fx * point_m[0] / z + camera.cx
    v = camera.fy * point_m[1] / z + camera.cy
    return int(round(u)), int(round(v))


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> List[float]:
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        z = float(np.arctan2(R[1, 0], R[0, 0]))
        y = float(np.arctan2(-R[2, 0], sy))
        x = float(np.arctan2(R[2, 1], R[2, 2]))
    else:
        z = 0.0
        y = float(np.arctan2(-R[2, 0], sy))
        x = float(np.arctan2(-R[1, 2], R[1, 1]))
    return [round(z, 6), round(y, 6), round(x, 6)]


def rack_frame_rotation() -> np.ndarray:
    """
    名义料架坐标系（对齐目标）。

    - **X 水平**：相机 +X
    - **Y 高度**：相机 -Y（图像向上）
    - **Z 深度**：相机 +Z（进架）
    """
    return np.column_stack([CAMERA_AXIS_X, CAMERA_UP, CAMERA_AXIS_Z])


def mask_major_axis_uv(mask: np.ndarray) -> np.ndarray:
    """mask 在图像上的主方向（u 右, v 下）。"""
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return np.array([1.0, 0.0], dtype=np.float64)
    du = xs.astype(np.float64) - float(xs.mean())
    dv = ys.astype(np.float64) - float(ys.mean())
    cov_uu = float((du * du).mean())
    cov_vv = float((dv * dv).mean())
    cov_uv = float((du * dv).mean())
    trace = cov_uu + cov_vv
    disc = max(trace * trace * 0.25 - (cov_uu * cov_vv - cov_uv * cov_uv), 0.0)
    lam_max = 0.5 * (trace + float(np.sqrt(disc)))
    if abs(cov_uv) < 1e-9:
        major = (
            np.array([1.0, 0.0], dtype=np.float64)
            if cov_uu >= cov_vv
            else np.array([0.0, 1.0], dtype=np.float64)
        )
    else:
        major = np.array([lam_max - cov_vv, cov_uv], dtype=np.float64)
    return major / (np.linalg.norm(major) + 1e-12)


def marker_horizontal_from_mask_3d(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
) -> np.ndarray:
    """
    用 mask 图像长轴 + 深度变化估计横杆方向（相机系）。

    斜拍时深度沿杆变化，3D PCA 会把进深混进主方向；2D 长轴更稳定。
    """
    u, v = mask_centroid(mask)
    major_uv = mask_major_axis_uv(mask)
    bx, by, bw, bh = mask_bbox_xywh(mask)
    span_px = max(12.0, 0.35 * max(float(bw), float(bh)))

    u1 = u + major_uv[0] * span_px
    v1 = v + major_uv[1] * span_px
    z0 = sample_depth_at(depth_m, u, v, mask=mask)
    z1 = sample_depth_at(depth_m, u1, v1, mask=mask)
    p0 = backproject(u, v, z0, camera)
    p1 = backproject(u1, v1, z1, camera)
    delta = p1 - p0
    dn = float(np.linalg.norm(delta))
    if dn < 1e-9:
        raise ValueError("无法从 mask 2D 长轴估计横杆方向")
    return delta / dn


def estimate_marker_surface_normal(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    marker 曲面法向：沿杆方向 × 向上（料架水平）。

    细长 mask 的 SVD 法向常近似沿杆方向，不能作为 Z 轴。
    """
    along_bar = marker_horizontal_from_mask_3d(mask, depth_m, camera)

    up = CAMERA_UP - float(np.dot(CAMERA_UP, along_bar)) * along_bar
    nu = float(np.linalg.norm(up))
    if nu < 1e-6:
        up = np.cross(along_bar, CAMERA_AXIS_Z)
        nu = float(np.linalg.norm(up))
    up = up / nu
    if float(np.dot(up, CAMERA_UP)) < 0:
        up = -up

    normal = np.cross(along_bar, up)
    nn = float(np.linalg.norm(normal))
    if nn < 1e-6:
        normal = np.cross(up, along_bar)
        nn = float(np.linalg.norm(normal))
    normal = normal / nn
    if float(np.dot(normal, CAMERA_AXIS_Z)) < 0:
        normal = -normal
    return normal, along_bar, up


def marker_rack_rotation_default_x_plane_normal(
    normal: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    marker 姿态约定：

    - **Z**：marker 平面法向
    - **X**：相机水平（+X_cam）在 ⊥Z 平面上的投影（绕法向转角）
    - **Y**：Z × X（右手系，对齐图像向上）
    """
    z_axis = normal / (np.linalg.norm(normal) + 1e-12)
    if z_axis[2] < 0:
        z_axis = -z_axis

    x_ref = CAMERA_AXIS_X
    x_axis = x_ref - float(np.dot(x_ref, z_axis)) * z_axis
    nx = float(np.linalg.norm(x_axis))
    if nx < 1e-6:
        x_axis = np.cross(CAMERA_UP, z_axis)
        nx = float(np.linalg.norm(x_axis))
    x_axis = x_axis / nx
    if float(np.dot(x_axis, x_ref)) < 0:
        x_axis = -x_axis

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    if float(np.dot(y_axis, CAMERA_UP)) < 0:
        y_axis = -y_axis
        x_axis = -x_axis

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    diagnostics = {
        "x_dot_default_horizontal": round(float(np.dot(x_axis, x_ref)), 4),
        "y_dot_camera_up": round(float(np.dot(y_axis, CAMERA_UP)), 4),
        "z_dot_plane_normal": round(float(np.dot(z_axis, normal / (np.linalg.norm(normal) + 1e-12))), 4),
    }
    return rotation, diagnostics


def marker_rack_rotation_horizontal_rack(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    料架水平 + 相机可斜拍：在相机坐标系下估计料架姿态。

    - X：mask 2D 长轴方向 + 深度抬升到 3D（沿横杆，物理水平）
    - Y：图像向上在 ⊥X 平面上的投影（料架高度 / 重力向上）
    - Z：X × Y，符号朝向场景深度（+Z_cam）
    """
    horizontal = marker_horizontal_from_mask_3d(mask, depth_m, camera)

    height = CAMERA_UP - float(np.dot(CAMERA_UP, horizontal)) * horizontal
    nh = float(np.linalg.norm(height))
    if nh < 1e-6:
        height = np.cross(horizontal, CAMERA_AXIS_Z)
        nh = float(np.linalg.norm(height))
        if nh < 1e-6:
            height = np.cross(CAMERA_AXIS_Z, horizontal)
            nh = float(np.linalg.norm(height))
    height = height / nh
    if float(np.dot(height, CAMERA_UP)) < 0:
        height = -height

    depth = np.cross(horizontal, height)
    depth = depth / (np.linalg.norm(depth) + 1e-12)
    if float(np.dot(depth, CAMERA_AXIS_Z)) < 0:
        horizontal = -horizontal
        depth = -depth

    height = np.cross(depth, horizontal)
    height = height / (np.linalg.norm(height) + 1e-12)

    rotation = np.column_stack([horizontal, height, depth])
    diagnostics = {
        "along_bar_dot_camera_x": round(float(np.dot(horizontal, CAMERA_AXIS_X)), 4),
        "height_dot_camera_up": round(float(np.dot(height, CAMERA_UP)), 4),
        "depth_dot_camera_z": round(float(np.dot(depth, CAMERA_AXIS_Z)), 4),
    }
    return rotation, diagnostics


def marker_rack_rotation_from_plane(normal: np.ndarray) -> np.ndarray:
    """
    从 marker 平面法向推算姿态，并对齐到料架轴。

    1. 用平面法向确定标记位所在平面
    2. Y = 料架高度轴（图像向上）在该平面上的投影
    3. Z = 料架深度轴（+Z_cam）在 ⊥Y 子空间中的方向
    4. X = Y×Z，符号对齐相机 +X（水平）
    """
    plane_n = normal / (np.linalg.norm(normal) + 1e-12)
    if plane_n[2] < 0:
        plane_n = -plane_n

    # 高度：料架 Y 在标记平面内的分量
    height = CAMERA_UP - float(np.dot(CAMERA_UP, plane_n)) * plane_n
    nh = float(np.linalg.norm(height))
    if nh < 1e-6:
        height = np.cross(plane_n, CAMERA_AXIS_X)
        nh = float(np.linalg.norm(height))
        if nh < 1e-6:
            height = CAMERA_AXIS_Y.copy()
            nh = 1.0
    height = height / nh

    # 深度：料架 +Z_cam，去除沿 height 的分量
    depth = CAMERA_AXIS_Z - float(np.dot(CAMERA_AXIS_Z, height)) * height
    nd = float(np.linalg.norm(depth))
    if nd < 1e-6:
        depth = np.cross(height, plane_n)
        nd = float(np.linalg.norm(depth))
    depth = depth / nd
    if float(np.dot(depth, CAMERA_AXIS_Z)) < 0:
        depth = -depth

    # 水平：右手系，符号对齐 +X_cam
    horizontal = np.cross(height, depth)
    horizontal = horizontal / (np.linalg.norm(horizontal) + 1e-12)
    if float(np.dot(horizontal, CAMERA_AXIS_X)) < 0:
        horizontal = -horizontal

    depth = np.cross(horizontal, height)
    depth = depth / (np.linalg.norm(depth) + 1e-12)
    if float(np.dot(depth, CAMERA_AXIS_Z)) < 0:
        horizontal = -horizontal
        depth = -depth

    return np.column_stack([horizontal, height, depth])


def rotation_from_normal(normal: np.ndarray) -> np.ndarray:
    return marker_rack_rotation_from_plane(normal)


def marker_rack_rotation(normal: np.ndarray) -> np.ndarray:
    return marker_rack_rotation_from_plane(normal)


def marker_horizontal_axis(rotation: np.ndarray) -> np.ndarray:
    """水平方向 = 局部 +X。"""
    return rotation[:, 0] / (np.linalg.norm(rotation[:, 0]) + 1e-12)


def marker_height_axis(rotation: np.ndarray) -> np.ndarray:
    """高度方向 = 局部 +Y。"""
    return rotation[:, 1] / (np.linalg.norm(rotation[:, 1]) + 1e-12)


def marker_depth_axis(rotation: np.ndarray) -> np.ndarray:
    """深度方向 = 局部 +Z。"""
    return rotation[:, 2] / (np.linalg.norm(rotation[:, 2]) + 1e-12)


def collect_mask_points_3d(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
    *,
    max_points: int = 2000,
) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("mask 为空")

    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points, dtype=int)
        ys, xs = ys[idx], xs[idx]

    points: List[np.ndarray] = []
    for y, x in zip(ys, xs):
        z = float(depth_m[y, x])
        if z <= 0 or not np.isfinite(z):
            continue
        points.append(backproject(float(x), float(y), z, camera))

    if len(points) < 10:
        u, v = mask_centroid(mask)
        z = sample_depth_at(depth_m, u, v, mask=mask)
        points.append(backproject(u, v, z, camera))

    if not points:
        raise ValueError("mask 区域无有效 3D 点")
    return np.asarray(points, dtype=np.float64)


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal / (np.linalg.norm(normal) + 1e-12)


def _order_square_corners_uv(pts: np.ndarray) -> np.ndarray:
    """将 4 个角点排序为 A(TL), B(TR), C(BR), D(BL)。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    a = pts[np.argmin(s)]
    c = pts[np.argmax(s)]
    b = pts[np.argmin(d)]
    d_pt = pts[np.argmax(d)]
    return np.array([a, b, c, d_pt], dtype=np.float64)


def _intersect_lines_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float64)


def detect_marker_square_corners_uv(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    从 mask 轮廓提取正方形四角 A,B,C,D（图像坐标 u,v）。

    失败时返回 None，由调用方回退质心方案。
    """
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) != 4:
        approx = cv2.approxPolyDP(contour, 0.08 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.float32)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2).astype(np.float64)
    return _order_square_corners_uv(pts)


def marker_center_uv_from_corners(corners_uv: np.ndarray) -> np.ndarray:
    """对角线 AC、BD 交点作为 p1 中心（与示意图一致）。"""
    a, b, c, d = corners_uv
    center = _intersect_lines_2d(a, c, b, d)
    if center is None:
        center = corners_uv.mean(axis=0)
    return center


def marker_aux_points_uv_from_corners(
    corners_uv: np.ndarray,
    center_uv: np.ndarray,
    *,
    diag_ratio: float = 0.5,
) -> np.ndarray:
    """
    沿对角线 p1→A/B/C/D 取辅助点 A',B',C',D'（默认在 50% 位置）。
    """
    ratio = float(np.clip(diag_ratio, 0.05, 0.95))
    aux: List[np.ndarray] = []
    for corner in corners_uv:
        aux.append(center_uv + ratio * (corner - center_uv))
    return np.asarray(aux, dtype=np.float64)


def backproject_uv_with_depth(
    uv: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
    *,
    depth_radius: int = 2,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    try:
        z = sample_depth_at(
            depth_m,
            float(uv[0]),
            float(uv[1]),
            radius=depth_radius,
            mask=mask,
        )
    except ValueError:
        return None
    return backproject(float(uv[0]), float(uv[1]), z, camera)


def marker_plane_normal_from_center_and_aux(
    center_3d: np.ndarray,
    aux_3d: np.ndarray,
) -> np.ndarray:
    """p1 + A'B'C'D' 五点拟合 marker 平面法向（Z 轴方向）。"""
    _, normal = fit_plane(np.vstack([center_3d.reshape(1, 3), aux_3d]))
    return normal


def marker_rotation_from_square_geometry(
    corners_3d: np.ndarray,
    *,
    plane_normal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    由正方形四角 3D 点确定姿态（示意图 ABCD）。

    - X：上下边平均方向 (B-A)+(C-D)
    - Y：左右边平均方向 (D-A)+(C-B)
    - Z：X×Y，朝向相机
    """
    a, b, c, d = corners_3d
    x_raw = (b - a) + (c - d)
    y_raw = (d - a) + (c - b)
    nx = float(np.linalg.norm(x_raw))
    ny = float(np.linalg.norm(y_raw))
    if nx < 1e-9 or ny < 1e-9:
        raise ValueError("正方形边方向退化")

    if plane_normal is not None:
        z_axis = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
        if z_axis[2] < 0:
            z_axis = -z_axis
    else:
        z_axis = np.cross(x_raw, y_raw)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
        if z_axis[2] < 0:
            z_axis = -z_axis

    x_axis = x_raw - float(np.dot(x_raw, z_axis)) * z_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    if float(np.dot(x_axis, CAMERA_AXIS_X)) < 0:
        x_axis = -x_axis

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    if float(np.dot(y_axis, CAMERA_UP)) < 0:
        y_axis = -y_axis
        x_axis = -x_axis

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    if z_axis[2] < 0:
        z_axis = -z_axis

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    diagnostics = {
        "x_dot_default_horizontal": round(float(np.dot(x_axis, CAMERA_AXIS_X)), 4),
        "y_dot_camera_up": round(float(np.dot(y_axis, CAMERA_UP)), 4),
        "z_dot_camera_z": round(float(np.dot(z_axis, CAMERA_AXIS_Z)), 4),
        "edge_x_mm": round(nx * 1000.0, 2),
        "edge_y_mm": round(ny * 1000.0, 2),
    }
    return rotation, diagnostics


def detect_circumscribed_square_corners_uv(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    圆形/近圆 mask 的最小外接圆 → 轴对齐外切正方形四角 A,B,C,D 与圆心 UV。

    :return: (corners_uv (4,2), center_uv (2,))
    """
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        center = np.asarray(mask_centroid(mask), dtype=np.float64)
        stats = mask_bbox_xywh(mask)
        x, y, w, h = stats
        r = 0.5 * max(float(w), float(h))
        corners = np.array(
            [
                [center[0] - r, center[1] - r],
                [center[0] + r, center[1] - r],
                [center[0] + r, center[1] + r],
                [center[0] - r, center[1] + r],
            ],
            dtype=np.float64,
        )
        return _order_square_corners_uv(corners), center

    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    center_uv = np.array([float(cx), float(cy)], dtype=np.float64)
    r = float(radius)
    corners = np.array(
        [
            [cx - r, cy - r],
            [cx + r, cy - r],
            [cx + r, cy + r],
            [cx - r, cy + r],
        ],
        dtype=np.float64,
    )
    return _order_square_corners_uv(corners), center_uv


def average_depth_at_uv_corners(
    depth_m: np.ndarray,
    corners_uv: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, List[float]]:
    """外接正方形四角 A–D 深度均值（P1 深度）。"""
    depths: List[float] = []
    for uv in corners_uv:
        z = sample_depth_at(depth_m, float(uv[0]), float(uv[1]), mask=mask)
        depths.append(z)
    return float(np.mean(depths)), depths


def shelf_horizontal_from_hole_centroids(
    centroids_uv: List[np.ndarray],
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """多个圆形孔洞中心 → 货架水平方向（相机系 3D 单位向量）。"""
    if len(centroids_uv) < 2:
        raise ValueError(f"至少需要 2 个孔洞中心，当前 {len(centroids_uv)}")

    ordered = sorted(centroids_uv, key=lambda p: float(p[0]))
    left_uv = np.asarray(ordered[0], dtype=np.float64)
    right_uv = np.asarray(ordered[-1], dtype=np.float64)
    z_left = sample_depth_at(depth_m, float(left_uv[0]), float(left_uv[1]))
    z_right = sample_depth_at(depth_m, float(right_uv[0]), float(right_uv[1]))
    p_left = backproject(float(left_uv[0]), float(left_uv[1]), z_left, camera)
    p_right = backproject(float(right_uv[0]), float(right_uv[1]), z_right, camera)
    delta = p_right - p_left
    dn = float(np.linalg.norm(delta))
    if dn < 1e-9:
        raise ValueError("孔洞中心 3D 方向退化")
    horizontal = delta / dn
    if float(np.dot(horizontal, CAMERA_AXIS_X)) < 0:
        horizontal = -horizontal
    return horizontal, {
        "hole_count": len(centroids_uv),
        "centroids_uv": [np.asarray(p, dtype=np.float64).round(2).tolist() for p in ordered],
        "left_uv": left_uv.round(2).tolist(),
        "right_uv": right_uv.round(2).tolist(),
        "span_px": round(float(np.linalg.norm(right_uv - left_uv)), 2),
        "horizontal_camera": horizontal.round(6).tolist(),
    }


def marker_rack_rotation_from_horizontal_and_plane(
    horizontal: np.ndarray,
    plane_normal: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    货架姿态：X=孔洞连线水平方向，Z=面板/LED 平面法向，Y=Z×X。
    """
    z_axis = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    if z_axis[2] < 0:
        z_axis = -z_axis

    x_axis = horizontal - float(np.dot(horizontal, z_axis)) * z_axis
    nx = float(np.linalg.norm(x_axis))
    if nx < 1e-6:
        x_axis = np.cross(CAMERA_UP, z_axis)
        nx = float(np.linalg.norm(x_axis))
    x_axis = x_axis / nx
    if float(np.dot(x_axis, horizontal)) < 0:
        x_axis = -x_axis

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    if float(np.dot(y_axis, CAMERA_UP)) < 0:
        y_axis = -y_axis
        x_axis = -x_axis

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    diagnostics = {
        "x_dot_holes_horizontal": round(float(np.dot(x_axis, horizontal)), 4),
        "y_dot_camera_up": round(float(np.dot(y_axis, CAMERA_UP)), 4),
        "z_dot_plane_normal": round(float(np.dot(z_axis, plane_normal / (np.linalg.norm(plane_normal) + 1e-12))), 4),
    }
    return rotation, diagnostics


def estimate_shelf_led_p1_pose(
    led_mask: np.ndarray,
    hole_masks: List[np.ndarray],
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
) -> Tuple[Pose6D, Dict[str, Any]]:
    """
    货架 P1：蓝色 LED 圆心为像素中心；深度=外接正方形四角均值；
    水平方向由面板圆形孔洞中心连线确定。
    """
    corners_uv, circle_center_uv = detect_circumscribed_square_corners_uv(led_mask)
    center_uv = np.asarray(circle_center_uv, dtype=np.float64)

    center_z, corner_depths = average_depth_at_uv_corners(
        depth_m, corners_uv, mask=led_mask
    )
    position = backproject(float(center_uv[0]), float(center_uv[1]), center_z, camera)

    corners_3d: List[np.ndarray] = []
    for uv, z in zip(corners_uv, corner_depths):
        corners_3d.append(backproject(float(uv[0]), float(uv[1]), z, camera))
    corners_arr = np.asarray(corners_3d, dtype=np.float64)
    _, plane_normal = fit_plane(corners_arr)

    hole_centroids: List[np.ndarray] = []
    hole_meta_list: List[Dict[str, Any]] = []
    for idx, hole_mask in enumerate(hole_masks, start=1):
        if not hole_mask.any():
            continue
        cu, cv = mask_centroid(hole_mask)
        hole_centroids.append(np.array([cu, cv], dtype=np.float64))
        hole_meta_list.append(
            {
                "id": idx,
                "centroid_uv": [round(cu, 2), round(cv, 2)],
                **mask_stats(hole_mask),
            }
        )

    rotation_method = "holes_horizontal_led_square_plane"
    hole_line_meta: Dict[str, Any] = {"holes_used": len(hole_centroids)}
    if len(hole_centroids) >= 2:
        horizontal, hole_line_meta = shelf_horizontal_from_hole_centroids(
            hole_centroids, depth_m, camera
        )
        rotation, rack_diag = marker_rack_rotation_from_horizontal_and_plane(
            horizontal, plane_normal
        )
    else:
        rotation, rack_diag = marker_rack_rotation_default_x_plane_normal(plane_normal)
        rotation_method = "led_square_plane_x_cam_fallback"
        hole_line_meta["fallback"] = "insufficient_holes"

    meta = {
        "method": "shelf_led_holes",
        "center_uv": center_uv.round(2).tolist(),
        "circle_center_uv": circle_center_uv.round(2).tolist(),
        "center_method": "led_min_enclosing_circle_center",
        "corners_uv": {
            "A": corners_uv[0].round(2).tolist(),
            "B": corners_uv[1].round(2).tolist(),
            "C": corners_uv[2].round(2).tolist(),
            "D": corners_uv[3].round(2).tolist(),
        },
        "corner_depths_m": [round(d, 6) for d in corner_depths],
        "depth_center_m": round(center_z, 6),
        "depth_method": "mean_of_square_corners_ABCD",
        "plane_normal": plane_normal.round(6).tolist(),
        "rotation_method": rotation_method,
        "rack_diagnostics": rack_diag,
        "holes": hole_meta_list,
        "hole_line": hole_line_meta,
        "p1_3d_mm": (position * 1000.0).round(2).tolist(),
    }
    return Pose6D(position_m=position, rotation=rotation), meta


def estimate_marker_pose(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraIntrinsics,
    *,
    aux_diag_ratio: float = 0.5,
) -> Tuple[Pose6D, Dict[str, Any]]:
    u_centroid, v_centroid = mask_centroid(mask)
    points = collect_mask_points_3d(mask, depth_m, camera)
    centroid_plane, normal_svd = fit_plane(points)
    surface_normal, along_bar, up_axis = estimate_marker_surface_normal(mask, depth_m, camera)

    corners_uv = detect_marker_square_corners_uv(mask)
    used_aux = False
    aux_meta: Dict[str, Any] = {"method": "centroid_fallback"}

    if corners_uv is not None:
        center_uv = marker_center_uv_from_corners(corners_uv)
        aux_uv = marker_aux_points_uv_from_corners(
            corners_uv, center_uv, diag_ratio=aux_diag_ratio
        )
        p1_3d = backproject_uv_with_depth(center_uv, depth_m, camera, depth_radius=3, mask=mask)
        if p1_3d is not None:
            aux_depths: List[float] = []
            aux_3d: List[np.ndarray] = []
            for aux in aux_uv:
                a3 = backproject_uv_with_depth(aux, depth_m, camera, depth_radius=3, mask=mask)
                if a3 is None:
                    aux_3d = []
                    break
                aux_3d.append(a3)
                aux_depths.append(float(a3[2]))

        if p1_3d is not None and len(aux_3d) == 4:
            aux_arr = np.asarray(aux_3d, dtype=np.float64)
            center_z = float(np.median([float(p1_3d[2])] + aux_depths))
            position = backproject(float(center_uv[0]), float(center_uv[1]), center_z, camera)
            plane_normal = marker_plane_normal_from_center_and_aux(position, aux_arr)
            rotation, rack_diag = marker_rack_rotation_default_x_plane_normal(plane_normal)
            used_aux = True
            aux_meta = {
                "method": "p1_aux_plane_normal_x_cam_horizontal",
                "diag_ratio": aux_diag_ratio,
                "corners_uv": {
                    "A": corners_uv[0].round(2).tolist(),
                    "B": corners_uv[1].round(2).tolist(),
                    "C": corners_uv[2].round(2).tolist(),
                    "D": corners_uv[3].round(2).tolist(),
                },
                "center_uv": center_uv.round(2).tolist(),
                "aux_uv": {
                    "A_prime": aux_uv[0].round(2).tolist(),
                    "B_prime": aux_uv[1].round(2).tolist(),
                    "C_prime": aux_uv[2].round(2).tolist(),
                    "D_prime": aux_uv[3].round(2).tolist(),
                },
                "p1_3d_mm": (position * 1000.0).round(2).tolist(),
                "aux_3d_mm": (aux_arr * 1000.0).round(2).tolist(),
                "plane_normal": plane_normal.round(6).tolist(),
                "depth_center_m": round(center_z, 6),
                "in_plane_rotation": "X = +X_cam projected onto plane perp Z",
            }

    if not used_aux:
        z_center = sample_depth_at(depth_m, u_centroid, v_centroid, mask=mask)
        position = backproject(u_centroid, v_centroid, z_center, camera)
        rotation, rack_diag = marker_rack_rotation_default_x_plane_normal(normal_svd)
        aux_meta["fallback_reason"] = "corner_or_aux_backproject_failed"
        aux_meta["plane_normal_source"] = "mask_svd"
    else:
        z_center = float(position[2])

    rack_ref = rack_frame_rotation()
    axis_alignment = {
        "X_dot_nominal": round(float(np.dot(rotation[:, 0], rack_ref[:, 0])), 4),
        "Y_dot_nominal": round(float(np.dot(rotation[:, 1], rack_ref[:, 1])), 4),
        "Z_dot_nominal": round(float(np.dot(rotation[:, 2], rack_ref[:, 2])), 4),
    }
    major_uv = mask_major_axis_uv(mask)

    meta = {
        "centroid_pixel": [round(u_centroid, 2), round(v_centroid, 2)],
        "plane_centroid_m": centroid_plane.round(6).tolist(),
        "plane_normal_svd": normal_svd.round(6).tolist(),
        "surface_normal": surface_normal.round(6).tolist(),
        "along_bar_camera": along_bar.round(6).tolist(),
        "up_axis_camera": up_axis.round(6).tolist(),
        "mask_major_axis_uv": [round(float(major_uv[0]), 4), round(float(major_uv[1]), 4)],
        "horizontal_axis_camera": rotation[:, 0].round(6).tolist(),
        "height_axis_camera": rotation[:, 1].round(6).tolist(),
        "normal_axis_camera": rotation[:, 2].round(6).tolist(),
        "rack_frame": (
            "Z=plane normal(p1+A'B'C'D'), X=+X_cam proj, Y=Z×X"
            if used_aux
            else "Z=mask SVD plane normal, X=+X_cam proj, Y=Z×X"
        ),
        "rotation_method": (
            "aux_plane_normal_x_cam_horizontal" if used_aux else "svd_plane_normal_x_cam_horizontal"
        ),
        "rack_diagnostics": rack_diag,
        "axis_alignment": axis_alignment,
        "num_3d_points": len(points),
        "depth_at_centroid_m": round(z_center, 6),
        "aux_reference": aux_meta,
    }
    return Pose6D(position_m=position, rotation=rotation), meta


def offset_pose(
    pose: Pose6D,
    *,
    depth_offset_mm: float,
    height_offset_mm: float,
) -> Pose6D:
    """
    在相机坐标系下偏移：深度沿 +Z，高度沿 -Y（图像向上）。

    用于 place 的 p2；new_grasp 请用 offset_pose_coplanar。
    """
    depth_m = depth_offset_mm / 1000.0
    height_m = height_offset_mm / 1000.0
    delta = depth_m * CAMERA_AXIS_Z + height_m * CAMERA_UP
    return Pose6D(position_m=pose.position_m + delta, rotation=pose.rotation.copy())


def offset_pose_coplanar(
    pose: Pose6D,
    *,
    depth_offset_mm: float,
    height_offset_mm: float,
) -> Pose6D:
    """
    在 p1 局部 YZ 平面内偏移（⊥局部 X），姿态不变。

    - height_offset → **+local Y**（料架高度）
    - depth_offset → **+local Z**（进架深度 / 法向轴）

    p1 / disk_center / p_grasp 三点共面且姿态相同。
    """
    depth_m = depth_offset_mm / 1000.0
    height_m = height_offset_mm / 1000.0
    y_ax = marker_height_axis(pose.rotation)
    z_ax = marker_depth_axis(pose.rotation)
    delta = height_m * y_ax + depth_m * z_ax
    return Pose6D(position_m=pose.position_m + delta, rotation=pose.rotation.copy())


def decompose_delta_in_yz_plane(
    rotation: np.ndarray,
    delta_m: np.ndarray,
) -> Tuple[float, float]:
    """将位移在局部 YZ 平面内分解为 Y、Z 分量（米）。"""
    y_ax = marker_height_axis(rotation)
    z_ax = marker_depth_axis(rotation)
    return float(np.dot(delta_m, y_ax)), float(np.dot(delta_m, z_ax))


def tray_rotation_camera() -> np.ndarray:
    """等价于 rack_frame_rotation。"""
    return rack_frame_rotation()


def tray_thickness_axis(rotation: np.ndarray) -> np.ndarray:
    """厚度方向 = 局部 +X（水平）。"""
    return marker_horizontal_axis(rotation)


def tray_diameter_axis(rotation: np.ndarray) -> np.ndarray:
    """直径竖直方向 = 局部 +Y（高度）。"""
    return marker_height_axis(rotation)


def sample_disk_points_3d(
    center_m: np.ndarray,
    rotation: np.ndarray,
    *,
    radius_m: float,
    num_samples: int = 72,
    face_normal_axis: int = 0,
) -> np.ndarray:
    """
    圆盘轮廓点。

    face_normal_axis=0：圆在 YZ 平面（法向 +X，place 薄边侧视）
    face_normal_axis=2：圆在 XY 平面（法向 +Z，盘面朝向相机）
    """
    angles = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    circle = np.zeros((num_samples, 3), dtype=np.float64)
    for i, angle in enumerate(angles):
        c, s = float(np.cos(angle)), float(np.sin(angle))
        if face_normal_axis == 2:
            local = np.array([radius_m * c, radius_m * s, 0.0], dtype=np.float64)
        else:
            local = np.array([0.0, radius_m * c, radius_m * s], dtype=np.float64)
        circle[i] = center_m + rotation @ local
    return circle


def sample_tray_edge_corners(
    center_m: np.ndarray,
    rotation: np.ndarray,
    *,
    radius_m: float,
    thickness_m: float,
) -> np.ndarray:
    """
    相机视角下料盘薄边矩形四角（近/远 × 上/下）。
    用于可视化侧视薄饼，而非圆盘面。
    """
    half_t = thickness_m * 0.5
    thick = tray_thickness_axis(rotation)
    diameter = tray_diameter_axis(rotation)
    top = center_m + radius_m * diameter
    bottom = center_m - radius_m * diameter
    return np.array(
        [
            top - half_t * thick,
            bottom - half_t * thick,
            bottom + half_t * thick,
            top + half_t * thick,
        ],
        dtype=np.float64,
    )


def sample_disk_thickness_segment(
    center_m: np.ndarray,
    rotation: np.ndarray,
    *,
    thickness_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """料盘厚度沿 X 方向（近端 / 远端）。"""
    half = thickness_m * 0.5
    axis = tray_thickness_axis(rotation)
    return center_m - half * axis, center_m + half * axis


def tray_rotation_yz_facing_camera(
    p1_rotation: np.ndarray,
    disk_center_m: np.ndarray,
) -> np.ndarray:
    """
    虚拟料盘姿态：YZ 平面朝向相机（盘面法向 +X 指向观察者）。

    - +X：盘心指向相机（视线方向），盘面平行于观察者
    - +Y：继承 p1 高度轴（图像向上分量）
    - +Z：+X × +Y，圆盘在 YZ 平面内
    """
    to_camera = -disk_center_m
    tc_norm = float(np.linalg.norm(to_camera))
    if tc_norm < 1e-9:
        x_axis = -CAMERA_AXIS_Z.copy()
    else:
        x_axis = to_camera / tc_norm

    y_ref = p1_rotation[:, 1]
    y_axis = y_ref - float(np.dot(y_ref, x_axis)) * x_axis
    ny = float(np.linalg.norm(y_axis))
    if ny < 1e-6:
        y_axis = CAMERA_UP - float(np.dot(CAMERA_UP, x_axis)) * x_axis
        ny = float(np.linalg.norm(y_axis))
    y_axis = y_axis / ny
    if float(np.dot(y_axis, CAMERA_UP)) < 0:
        y_axis = -y_axis

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    return np.column_stack([x_axis, y_axis, z_axis])


def disk_center_pose_from_p1(
    p1: Pose6D,
    *,
    depth_offset_mm: float,
    height_offset_mm: float,
) -> Pose6D:
    """p1 在局部 YZ 平面内偏移得圆盘中心，姿态与 p1 相同。"""
    return offset_pose_coplanar(
        p1,
        depth_offset_mm=depth_offset_mm,
        height_offset_mm=height_offset_mm,
    )


def compute_grasp_from_disk_center(
    disk_center: Pose6D,
    *,
    tray_diameter_mm: float,
) -> Pose6D:
    """
    抓取点：圆盘中心在 YZ 平面内，沿局部 **-Z** 偏移半径 R。

    姿态与 disk_center 相同；与 p1、disk_center 共面。
    """
    radius_m = tray_diameter_mm / 2000.0
    z_ax = marker_depth_axis(disk_center.rotation)
    position = disk_center.position_m - radius_m * z_ax
    return Pose6D(position_m=position, rotation=disk_center.rotation.copy())


def compute_place_target_from_p2(
    p2: Pose6D,
    *,
    tray_diameter_mm: float,
) -> Pose6D:
    """
    放置目标点：从圆盘中心 p2 沿局部 **-X 水平** 偏移料盘半径。

    对应示意图圆盘左侧边缘。
    """
    radius_m = tray_diameter_mm / 2000.0
    axis_horizontal = marker_horizontal_axis(p2.rotation)
    position = p2.position_m - radius_m * axis_horizontal
    return Pose6D(position_m=position, rotation=p2.rotation.copy())


def mask_bbox_xywh(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def mask_stats(mask: np.ndarray) -> Dict[str, Any]:
    ys, xs = np.where(mask)
    x, y, w, h = mask_bbox_xywh(mask)
    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
    return {
        "foreground_pixels": int(len(xs)),
        "bbox_xywh": [x, y, w, h],
        "aspect_ratio": round(aspect, 4),
        "centroid_pixel": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
    }
