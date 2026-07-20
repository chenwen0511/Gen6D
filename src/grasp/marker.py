"""货架面板孔洞 + 蓝色 LED 标记位筛选与 P1 位姿可视化。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.grasp.paths import PROMPT_DIR
from src.grasp.place_geometry import (
    CameraIntrinsics,
    Pose6D,
    estimate_shelf_led_p1_pose,
    load_camera_json,
    mask_centroid,
    mask_stats,
    project_point,
)
from src.grasp.sam3 import _decode_detection_mask, infer_sam3_with_image
from src.grasp.settings import (
    DEFAULT_PLACE_HOLE_PROMPT,
    DEFAULT_PLACE_LED_PROMPT,
    DEFAULT_PLACE_MARKER_PROMPT,
    DEFAULT_PLACE_SAM3_MASK_THRESHOLD,
    DEFAULT_PLACE_SAM3_THRESHOLD,
)

DEFAULT_HOLE_PROMPT_FILE = "shelf_holes.txt"
DEFAULT_LED_PROMPT_FILE = "blue_led_marker.txt"
DEFAULT_MARKER_PROMPT_FILE = DEFAULT_LED_PROMPT_FILE


def load_marker_prompt(filename: str = DEFAULT_LED_PROMPT_FILE) -> str:
    return load_led_prompt(filename)


def load_hole_prompt(filename: str = DEFAULT_HOLE_PROMPT_FILE) -> str:
    path = PROMPT_DIR / filename
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_PLACE_HOLE_PROMPT


def load_led_prompt(filename: str = DEFAULT_LED_PROMPT_FILE) -> str:
    path = PROMPT_DIR / filename
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_PLACE_LED_PROMPT or DEFAULT_PLACE_MARKER_PROMPT


def _blue_score(rgb: Image.Image, mask: np.ndarray) -> float:
    arr = np.array(rgb.convert("RGB"))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    pixels = arr[ys, xs].astype(np.float32)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    blueness = b - 0.5 * (r + g)
    return float(np.clip(blueness.mean() / 128.0, 0.0, 1.0))


def select_blue_led_detection(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    rgb: Image.Image,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """按蓝度 × 圆度 × score 选出发光蓝色 LED 圆形标记。"""
    if not detections:
        return None, {"reason": "empty"}

    width, height = image_size
    candidates = []
    for idx, det in enumerate(detections, start=1):
        mask = _decode_detection_mask(det, image_size)
        stats = mask_stats(mask)
        aspect = float(stats["aspect_ratio"])
        if aspect < 0.55:
            continue
        if stats["foreground_pixels"] < 12:
            continue
        bbox = stats["bbox_xywh"]
        bw, bh = bbox[2], bbox[3]
        if max(bw, bh) > min(width, height) * 0.2:
            continue
        blue = _blue_score(rgb, mask)
        if blue < 0.08:
            continue
        score = float(det.get("score", 0.0)) * (0.4 + 0.6 * aspect) * (0.2 + 0.8 * blue)
        candidates.append(
            {
                "id": idx,
                "score": round(score, 4),
                "sam3_score": float(det.get("score", 0.0)),
                "blue_score": round(blue, 4),
                "aspect_ratio": aspect,
                "bbox_xywh": bbox,
            }
        )

    if not candidates:
        best_idx = int(np.argmax([float(d.get("score", 0.0)) for d in detections]))
        return detections[best_idx], {"fallback": "highest_sam3_score", "candidates": []}

    best = max(candidates, key=lambda c: c["score"])
    return detections[best["id"] - 1], {"selected": best, "candidates": candidates}


def select_hole_detections(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """保留货架面板上有效的圆形孔洞检测（按 u 从左到右）。"""
    if not detections:
        return [], {"reason": "empty"}

    width, height = image_size
    kept: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for idx, det in enumerate(detections, start=1):
        mask = _decode_detection_mask(det, image_size)
        stats = mask_stats(mask)
        aspect = float(stats["aspect_ratio"])
        if aspect < 0.45:
            continue
        if stats["foreground_pixels"] < 8:
            continue
        bbox = stats["bbox_xywh"]
        bw, bh = bbox[2], bbox[3]
        if max(bw, bh) > min(width, height) * 0.15:
            continue
        cu = float(stats["centroid_pixel"][0])
        meta = {
            "id": idx,
            "sam3_score": float(det.get("score", 0.0)),
            "aspect_ratio": aspect,
            "bbox_xywh": bbox,
            "centroid_uv": stats["centroid_pixel"],
        }
        kept.append((cu, det, meta))

    kept.sort(key=lambda item: item[0])
    selected = [item[1] for item in kept]
    candidates = [item[2] for item in kept]
    return selected, {"count": len(selected), "candidates": candidates}


def select_marker_detection(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    rgb: Image.Image,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """兼容旧名：等同 select_blue_led_detection。"""
    return select_blue_led_detection(detections, image_size, rgb)


def depth_mm_to_meters(depth_mm: np.ndarray) -> np.ndarray:
    """Gen6D 传感器深度（mm）→ meters。"""
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


def render_shelf_p1_visualization(
    rgb: Image.Image,
    *,
    led_mask: np.ndarray,
    hole_masks: List[np.ndarray],
    p1: Pose6D,
    camera: CameraIntrinsics,
    pose_meta: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """蓝色 LED + 面板孔洞 + 外接正方形 ABCD + P1 坐标轴。"""
    bgr = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    for hole_mask in hole_masks:
        tint = np.zeros_like(overlay)
        tint[hole_mask] = (180, 180, 180)
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0)
        cu, cv = mask_centroid(hole_mask)
        cv2.circle(overlay, (int(round(cu)), int(round(cv))), 4, (120, 120, 120), 2, cv2.LINE_AA)

    led_tint = np.zeros_like(overlay)
    led_tint[led_mask] = (255, 120, 0)
    overlay = cv2.addWeighted(overlay, 1.0, led_tint, 0.45, 0)

    center_uv = np.asarray((pose_meta or {}).get("center_uv") or mask_centroid(led_mask), dtype=np.float64)
    cu = (int(round(center_uv[0])), int(round(center_uv[1])))
    cv2.drawMarker(overlay, cu, (255, 120, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)

    corners_uv = (pose_meta or {}).get("corners_uv") or {}
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
        cv2.polylines(overlay, [np.array(corner_pts, dtype=np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)

    hole_line = (pose_meta or {}).get("hole_line") or {}
    left_uv = hole_line.get("left_uv")
    right_uv = hole_line.get("right_uv")
    if left_uv and right_uv:
        line_p0 = (int(round(left_uv[0])), int(round(left_uv[1])))
        line_p1 = (int(round(right_uv[0])), int(round(right_uv[1])))
        cv2.line(overlay, line_p0, line_p1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay, "holes H", (line_p0[0], line_p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
        )

    p1_uv = project_point(p1.position_m, camera)
    if p1_uv:
        cv2.drawMarker(
            overlay, p1_uv, (255, 220, 0), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=3
        )
        cv2.circle(overlay, p1_uv, 10, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            "P1",
            (p1_uv[0] + 12, p1_uv[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 220, 0),
            2,
            cv2.LINE_AA,
        )
        _draw_axes(overlay, p1, camera, scale_m=0.045)

    pos_mm = p1.position_mm.round(1).tolist()
    rot_method = (pose_meta or {}).get("rotation_method", "unknown")
    hole_count = len(hole_masks)
    _draw_legend(
        overlay,
        [
            "shelf P1 (blue LED + holes)",
            f"P1 position_mm: {pos_mm}",
            f"depth: mean ABCD corners",
            f"holes: {hole_count}, rotation: {rot_method}",
            "X=hole horizontal, Z=LED plane, Y=Z×X",
        ],
    )
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def render_marker_p1_visualization(
    rgb: Image.Image,
    *,
    marker_mask: np.ndarray,
    p1: Pose6D,
    camera: CameraIntrinsics,
    pose_meta: Optional[Dict[str, Any]] = None,
    hole_masks: Optional[List[np.ndarray]] = None,
) -> Image.Image:
    return render_shelf_p1_visualization(
        rgb,
        led_mask=marker_mask,
        hole_masks=hole_masks or [],
        p1=p1,
        camera=camera,
        pose_meta=pose_meta,
    )


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
    """在 RGB（或分割叠加图）上标注 P1 与一组实例点。"""
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


def estimate_p1_from_shelf_markers(
    rgb: Image.Image,
    led_mask: np.ndarray,
    hole_masks: List[np.ndarray],
    depth_mm: np.ndarray,
    camera: CameraIntrinsics,
) -> Tuple[Pose6D, Dict[str, Any], Image.Image]:
    """面板孔洞 + 蓝色 LED → P1 位姿 + 可视化。

    规则：
    1. 所有货架面板圆孔用于确定货架水平方向；
    2. 蓝色 LED 圆心作为 P1 像素中心；
    3. LED 外切正方形四角 A/B/C/D 深度均值作为 P1 深度。
    """
    depth_m = depth_mm_to_meters(depth_mm)
    p1, meta = estimate_shelf_led_p1_pose(led_mask, hole_masks, depth_m, camera)
    vis = render_shelf_p1_visualization(
        rgb,
        led_mask=led_mask,
        hole_masks=hole_masks,
        p1=p1,
        camera=camera,
        pose_meta=meta,
    )
    return p1, meta, vis


def estimate_p1_from_marker(
    rgb: Image.Image,
    marker_mask: np.ndarray,
    depth_mm: np.ndarray,
    camera: CameraIntrinsics,
    *,
    hole_masks: Optional[List[np.ndarray]] = None,
) -> Tuple[Pose6D, Dict[str, Any], Image.Image]:
    """兼容旧接口：marker_mask 视为 LED mask。"""
    return estimate_p1_from_shelf_markers(
        rgb, marker_mask, hole_masks or [], depth_mm, camera
    )


def infer_p1_from_shelf_panel(
    rgb: Image.Image,
    depth_mm: np.ndarray,
    camera: CameraIntrinsics,
    image_size: Tuple[int, int],
    *,
    hole_prompt: str,
    led_prompt: str,
    api_url: str,
    threshold: float = DEFAULT_PLACE_SAM3_THRESHOLD,
    mask_threshold: float = DEFAULT_PLACE_SAM3_MASK_THRESHOLD,
    timeout_s: float = 300.0,
) -> Tuple[Optional[Pose6D], Dict[str, Any], Optional[Image.Image]]:
    """
    两次 SAM3：面板孔洞定水平 + 蓝色 LED 定 P1。
    返回 (p1_pose|None, marker_payload, p1_vis|None)。
    """
    payload: Dict[str, Any] = {
        "hole_prompt": hole_prompt,
        "led_prompt": led_prompt,
    }
    hole_masks: List[np.ndarray] = []
    led_mask: Optional[np.ndarray] = None

    t0 = time.perf_counter()
    hole_result, _ = infer_sam3_with_image(
        rgb,
        prompt=hole_prompt,
        api_url=api_url,
        threshold=float(threshold),
        mask_threshold=float(mask_threshold),
        timeout_s=float(timeout_s),
        return_vis_base64=True,
        filter_by_point=False,
    )
    payload["holes"] = {
        "sam3_elapsed_s": round(time.perf_counter() - t0, 3),
        "num_detections": hole_result.num_detections,
    }
    hole_dets, hole_selection = select_hole_detections(
        list(hole_result.detections or []), image_size
    )
    payload["holes"]["selection"] = hole_selection
    for det in hole_dets:
        hole_masks.append(_decode_detection_mask(det, image_size))

    t1 = time.perf_counter()
    led_result, _ = infer_sam3_with_image(
        rgb,
        prompt=led_prompt,
        api_url=api_url,
        threshold=float(threshold),
        mask_threshold=float(mask_threshold),
        timeout_s=float(timeout_s),
        return_vis_base64=True,
        filter_by_point=False,
    )
    payload["led"] = {
        "sam3_elapsed_s": round(time.perf_counter() - t1, 3),
        "num_detections": led_result.num_detections,
    }

    if not led_result.detections:
        payload["success"] = False
        payload["message"] = "SAM3 未检测到蓝色 LED 标记"
        return None, payload, None

    led_det, led_selection = select_blue_led_detection(
        list(led_result.detections), image_size, rgb
    )
    payload["led"]["selection"] = led_selection
    if led_det is None:
        payload["success"] = False
        payload["message"] = "未能选出蓝色 LED 标记"
        return None, payload, None

    led_mask = _decode_detection_mask(led_det, image_size)
    p1, p1_meta, vis = estimate_p1_from_shelf_markers(
        rgb, led_mask, hole_masks, depth_mm, camera
    )
    payload["success"] = True
    payload["p1"] = p1.to_dict()
    payload["p1_meta"] = p1_meta
    payload["num_holes"] = len(hole_masks)
    return p1, payload, vis
