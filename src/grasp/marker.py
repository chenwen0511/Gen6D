"""绿色标记位筛选与 P1 位姿可视化（移植自 PEM_service place / new_grasp）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.grasp.paths import PROMPT_DIR
from src.grasp.place_geometry import (
    CameraIntrinsics,
    Pose6D,
    estimate_marker_pose,
    load_camera_json,
    load_depth_meters,
    mask_centroid,
    mask_stats,
    project_point,
)
from src.grasp.sam3 import _decode_detection_mask
from src.grasp.settings import DEFAULT_PLACE_MARKER_PROMPT

DEFAULT_MARKER_PROMPT_FILE = "green_square_marker.txt"


def load_marker_prompt(filename: str = DEFAULT_MARKER_PROMPT_FILE) -> str:
    path = PROMPT_DIR / filename
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_PLACE_MARKER_PROMPT


def _green_score(rgb: Image.Image, mask: np.ndarray) -> float:
    arr = np.array(rgb.convert("RGB"))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    pixels = arr[ys, xs].astype(np.float32)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    greenness = g - 0.5 * (r + b)
    return float(np.clip(greenness.mean() / 128.0, 0.0, 1.0))


def select_marker_detection(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    rgb: Image.Image,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """按绿度 × 长宽比 × score 选出绿色方形标记位。"""
    if not detections:
        return None, {"reason": "empty"}

    width, height = image_size
    candidates = []
    for idx, det in enumerate(detections, start=1):
        mask = _decode_detection_mask(det, image_size)
        stats = mask_stats(mask)
        aspect = float(stats["aspect_ratio"])
        if aspect < 0.45:
            continue
        if stats["foreground_pixels"] < 20:
            continue
        bbox = stats["bbox_xywh"]
        bw, bh = bbox[2], bbox[3]
        if max(bw, bh) > min(width, height) * 0.25:
            continue
        green = _green_score(rgb, mask)
        score = float(det.get("score", 0.0)) * (0.5 + 0.5 * aspect) * (0.3 + 0.7 * green)
        candidates.append(
            {
                "id": idx,
                "score": round(score, 4),
                "sam3_score": float(det.get("score", 0.0)),
                "green_score": round(green, 4),
                "aspect_ratio": aspect,
                "bbox_xywh": bbox,
            }
        )

    if not candidates:
        best_idx = int(np.argmax([float(d.get("score", 0.0)) for d in detections]))
        return detections[best_idx], {"fallback": "highest_sam3_score", "candidates": []}

    best = max(candidates, key=lambda c: c["score"])
    return detections[best["id"] - 1], {"selected": best, "candidates": candidates}


def depth_mm_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """Gen6D 传感器深度（mm）→ meters，供 estimate_marker_pose 使用。"""
    depth_f = depth_mm.astype(np.float64)
    depth_m = depth_f.copy()
    valid = np.isfinite(depth_f) & (depth_f > 0)
    depth_m[~valid] = 0.0
    if valid.any() and float(np.median(depth_f[valid])) > 5.0:
        depth_m[valid] = depth_f[valid] / 1000.0
    return depth_m


def camera_from_path(camera_path: Path) -> CameraIntrinsics:
    return load_camera_json(camera_path)


def _draw_axes(
    bgr: np.ndarray,
    pose: Pose6D,
    camera: CameraIntrinsics,
    *,
    scale_m: float = 0.045,
) -> None:
    origin = pose.position_m
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labels = ["X", "Y", "Z"]
    for idx, color in enumerate(colors):
        end = origin + scale_m * pose.rotation[:, idx]
        p0 = project_point(origin, camera)
        p1 = project_point(end, camera)
        if p0 and p1:
            cv2.line(bgr, p0, p1, color, 2, cv2.LINE_AA)
            cv2.putText(
                bgr,
                labels[idx],
                (p1[0] + 4, p1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )


def _draw_legend(bgr: np.ndarray, lines: List[str]) -> None:
    y = 24
    for text in lines:
        cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        y += 22


def render_marker_p1_visualization(
    rgb: Image.Image,
    *,
    marker_mask: np.ndarray,
    p1: Pose6D,
    camera: CameraIntrinsics,
    pose_meta: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """
    参考 PEM new_grasp Step2：绿色 mask、四角 A–D、辅助点 A'–D'、p1 十字 + 坐标轴。
    """
    bgr = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    mask_color = np.zeros_like(overlay)
    mask_color[marker_mask] = (0, 220, 0)
    overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.35, 0)

    centroid_uv = mask_centroid(marker_mask)
    cu = (int(round(centroid_uv[0])), int(round(centroid_uv[1])))
    cv2.drawMarker(overlay, cu, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)

    p1_uv = project_point(p1.position_m, camera)
    if p1_uv:
        cv2.drawMarker(
            overlay, p1_uv, (255, 220, 0), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=3
        )
        cv2.circle(overlay, p1_uv, 10, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            "p1",
            (p1_uv[0] + 12, p1_uv[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )
        _draw_axes(overlay, p1, camera, scale_m=0.045)
        cv2.line(overlay, cu, p1_uv, (200, 200, 200), 1, cv2.LINE_AA)

    aux_ref = (pose_meta or {}).get("aux_reference") or {}
    corners_uv = aux_ref.get("corners_uv") or {}
    aux_uv_map = aux_ref.get("aux_uv") or {}
    corner_pts: List[Tuple[int, int]] = []
    for key, label in zip(("A", "B", "C", "D"), ("A", "B", "C", "D")):
        uv = corners_uv.get(key)
        if not uv:
            continue
        pt = (int(round(uv[0])), int(round(uv[1])))
        corner_pts.append(pt)
        cv2.circle(overlay, pt, 5, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay, label, (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA
        )
    if len(corner_pts) == 4:
        cv2.polylines(overlay, [np.array(corner_pts, dtype=np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)

    aux_pts: List[Tuple[int, int]] = []
    for key, label in zip(
        ("A_prime", "B_prime", "C_prime", "D_prime"),
        ("A'", "B'", "C'", "D'"),
    ):
        uv = aux_uv_map.get(key)
        if not uv:
            continue
        pt = (int(round(uv[0])), int(round(uv[1])))
        aux_pts.append(pt)
        cv2.circle(overlay, pt, 7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay, label, (pt[0] + 8, pt[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA
        )
        if p1_uv:
            cv2.line(overlay, p1_uv, pt, (140, 140, 140), 2, cv2.LINE_AA)
    if len(aux_pts) == 4:
        cv2.polylines(overlay, [np.array(aux_pts, dtype=np.int32)], True, (0, 200, 255), 2, cv2.LINE_AA)

    pos_mm = p1.position_mm.round(1).tolist()
    rot_method = (pose_meta or {}).get("rotation_method", "unknown")
    if p1_uv:
        du = p1_uv[0] - cu[0]
        dv = p1_uv[1] - cu[1]
        du_text = f"p1 - centroid px: ({du:+.1f}, {dv:+.1f})"
    else:
        du_text = "p1 - centroid px: n/a"
    _draw_legend(
        overlay,
        [
            "marker pose p1",
            f"p1 position_mm: {pos_mm}",
            du_text,
            f"rotation: {rot_method}",
            "Z=marker plane, X=camera horizontal on plane, Y=ZxX",
        ],
    )
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def render_p1_and_pi_preview(
    rgb: Image.Image,
    camera: CameraIntrinsics,
    *,
    p1: Optional[Pose6D] = None,
    pi_list: Optional[List[Dict[str, Any]]] = None,
    rotation_fallback: Optional[np.ndarray] = None,
    title: str = "P1 / p_i preview",
    point_prefix: str = "p",
    dist_key: str = "z_mm",
    dist_label: str = "z",
    draw_link_to_p1: bool = False,
) -> Image.Image:
    """
    在 RGB（或分割叠加图）上标注 P1 与一组实例点。
    列表项需含 instance_id、position_mm；可选 dist_key 写入图例。
    """
    bgr = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
    pi_bgr_colors = (
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (128, 255, 0),
        (128, 64, 255),
    )
    legend = [title]
    p1_uv = None

    if p1 is not None:
        p1_uv = project_point(p1.position_m, camera)
        if p1_uv:
            cv2.drawMarker(
                bgr, p1_uv, (0, 220, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=3
            )
            cv2.circle(bgr, p1_uv, 14, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(
                bgr,
                "P1",
                (p1_uv[0] + 14, p1_uv[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            _draw_axes(bgr, p1, camera, scale_m=0.05)
        pos_mm = p1.position_mm.round(1).tolist()
        legend.append(f"P1 mm: {pos_mm}")

    rot = None
    if p1 is not None:
        rot = p1.rotation
    elif rotation_fallback is not None:
        rot = np.asarray(rotation_fallback, dtype=np.float64)

    for item in pi_list or []:
        inst_id = int(item.get("instance_id", 0))
        pos_mm = np.asarray(item.get("position_mm"), dtype=np.float64)
        if pos_mm.shape != (3,):
            continue
        pose_i = Pose6D(
            position_m=pos_mm / 1000.0,
            rotation=rot if rot is not None else np.eye(3, dtype=np.float64),
        )
        uv = project_point(pose_i.position_m, camera)
        if not uv:
            continue
        color = pi_bgr_colors[(inst_id - 1) % len(pi_bgr_colors)]
        if draw_link_to_p1 and p1_uv is not None:
            cv2.line(bgr, p1_uv, uv, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.drawMarker(
            bgr, uv, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=24, thickness=3
        )
        cv2.circle(bgr, uv, 12, color, 2, cv2.LINE_AA)
        label = f"{point_prefix}{inst_id}"
        cv2.putText(
            bgr,
            label,
            (uv[0] + 12, uv[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        _draw_axes(bgr, pose_i, camera, scale_m=0.06)
        if dist_key in item:
            legend.append(f"{label} {dist_label}={item[dist_key]}")
        else:
            legend.append(f"{label} z={round(float(pos_mm[2]), 1)}")

    if len(legend) == 1:
        legend.append("no points")
    _draw_legend(bgr, legend)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def estimate_p1_from_marker(
    rgb: Image.Image,
    marker_mask: np.ndarray,
    depth_mm: np.ndarray,
    camera: CameraIntrinsics,
) -> Tuple[Pose6D, Dict[str, Any], Image.Image]:
    """标记 mask + 传感器深度 → p1 位姿 + 可视化。"""
    depth_m = depth_mm_to_meters(depth_mm)
    p1, meta = estimate_marker_pose(marker_mask, depth_m, camera)
    vis = render_marker_p1_visualization(
        rgb,
        marker_mask=marker_mask,
        p1=p1,
        camera=camera,
        pose_meta=meta,
    )
    return p1, meta, vis
