"""抓取点 q 推理：与「抓取 位姿估计」Tab 同逻辑，供 REST / UI 复用。"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.depth.pointcloud import (
    annotate_p1_x_distance,
    find_instance_qi_from_pi_sphere,
    select_nearest_along_p1_x,
)
from src.depth.service import load_intrinsics_from_dict, load_sensor_depth_from_image
from src.grasp.marker import (
    estimate_p1_from_marker,
    load_marker_prompt,
    render_p1_and_pi_preview,
    select_marker_detection,
)
from src.grasp.place_geometry import CameraIntrinsics, Pose6D, rotation_matrix_to_euler_zyx
from src.grasp.sam3 import (
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
    _decode_detection_mask,
    build_instance_id_mask_from_detections,
    decode_sam3_raw_visualization,
    infer_sam3_with_image,
    render_sam3_mask_bbox_previews,
)
from src.grasp.settings import (
    DEFAULT_PLACE_MARKER_PROMPT,
    DEFAULT_PLACE_SAM3_MASK_THRESHOLD,
    DEFAULT_PLACE_SAM3_THRESHOLD,
)

logger = logging.getLogger("grasp_infer")


def format_grasp_xyzrxryrz(
    position_mm: Any,
    rotation_3x3: Any,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    夹爪抓取位姿：``[x, y, z, rx, ry, rz]``。
    xyz 单位 mm；rx/ry/rz 为 °（旋转矩阵按 ZYX 欧拉角换算）。
    """
    pos = np.asarray(position_mm, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation_3x3, dtype=np.float64).reshape(3, 3)
    z_rad, y_rad, x_rad = rotation_matrix_to_euler_zyx(rot)
    rx = round(float(np.degrees(x_rad)), 3)
    ry = round(float(np.degrees(y_rad)), 3)
    rz = round(float(np.degrees(z_rad)), 3)
    xyz = [round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2)]
    xyzrxryrz = [xyz[0], xyz[1], xyz[2], rx, ry, rz]
    out: Dict[str, Any] = {
        "success": True,
        "xyzrxryrz": xyzrxryrz,
        "unit": {"xyz": "mm", "rx_ry_rz": "deg"},
        "euler_convention": "ZYX (rz,ry,rx) → displayed as [x,y,z,rx,ry,rz]",
        "position_mm": xyz,
        "rpy_deg": {"rx": rx, "ry": ry, "rz": rz},
        "rotation_matrix": rot.round(6).tolist(),
    }
    if meta:
        out["meta"] = meta
    return out


def _detection_summary(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, det in enumerate(detections, start=1):
        rows.append({"id": idx, "score": float(det.get("score", 0.0)), "bbox": det.get("bbox")})
    return rows


@dataclass
class GraspInferResult:
    success: bool
    message: str = ""
    xyzrxryrz: Optional[List[float]] = None
    grasp_pose: Optional[Dict[str, Any]] = None
    grasp_vis: Optional[Image.Image] = None
    mask_vis: Optional[Image.Image] = None
    bbox_vis: Optional[Image.Image] = None
    p1_vis: Optional[Image.Image] = None
    elapsed_s: float = 0.0
    num_instances: int = 0
    marker_p1: Dict[str, Any] = field(default_factory=dict)
    instance_qi: List[Dict[str, Any]] = field(default_factory=list)
    instance_qi_all: List[Dict[str, Any]] = field(default_factory=list)
    image_size: Optional[List[int]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> Dict[str, Any]:
        """供 REST 返回（不含图片二进制；图片单独塞 images）。"""
        body: Dict[str, Any] = {
            "success": self.success,
            "message": self.message,
            "elapsed_s": self.elapsed_s,
            "xyzrxryrz": self.xyzrxryrz,
            "unit": {"xyz": "mm", "rx_ry_rz": "deg"},
            "grasp_pose": self.grasp_pose,
            "num_instances": self.num_instances,
            "marker_p1": self.marker_p1,
            "instance_qi": self.instance_qi,
            "instance_qi_all": self.instance_qi_all,
            "image_size": self.image_size,
        }
        if self.extras:
            body["extras"] = self.extras
        return body


def infer_grasp(
    rgb: Image.Image,
    sensor_depth: np.ndarray,
    place_camera: CameraIntrinsics,
    intrinsics: np.ndarray,
    *,
    prompt: Optional[str] = None,
    marker_prompt: Optional[str] = None,
    enable_marker_p1: bool = True,
    api_url: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    timeout_s: Optional[float] = None,
    y_band_mm: float = 8.0,
    radius_mm: float | None = None,
) -> GraspInferResult:
    """
    与「抓取 位姿估计」页签同款流水线：实例分割 → P1 → q_i → 沿 P1-X 最近 q。

    :param sensor_depth: 传感器深度，单位 mm，形状 (H, W)
    :param intrinsics: 3×3 cam_K
    """
    t0 = time.perf_counter()
    prompt_text = (prompt or DEFAULT_SAM3_PROMPT or "").strip()
    if not prompt_text:
        return GraspInferResult(success=False, message="实例分割提示词不能为空")

    depth_h, depth_w = sensor_depth.shape
    rgb_w, rgb_h = rgb.size
    if (rgb_w, rgb_h) != (depth_w, depth_h):
        image_for_seg = rgb.convert("RGB").resize((depth_w, depth_h), Image.BILINEAR)
        logger.warning(
            "RGB %sx%s ≠ depth %sx%s，已 resize 到 depth 尺寸",
            rgb_w,
            rgb_h,
            depth_w,
            depth_h,
        )
    else:
        image_for_seg = rgb.convert("RGB")

    sam_api = (api_url or "").strip() or DEFAULT_SAM3_API_URL
    thr = float(threshold if threshold is not None else DEFAULT_SAM3_THRESHOLD)
    mask_thr = float(mask_threshold if mask_threshold is not None else DEFAULT_SAM3_MASK_THRESHOLD)
    timeout = float(timeout_s if timeout_s is not None else DEFAULT_SAM3_TIMEOUT_S)

    # --- 1) 实例分割 ---
    try:
        result, vis = infer_sam3_with_image(
            image_for_seg,
            prompt=prompt_text,
            api_url=sam_api,
            threshold=thr,
            mask_threshold=mask_thr,
            timeout_s=timeout,
            return_vis_base64=True,
            filter_by_point=False,
        )
    except Exception as exc:
        logger.exception("sam3 instance failed")
        return GraspInferResult(
            success=False,
            message=f"SAM3 实例分割失败: {exc}",
            elapsed_s=round(time.perf_counter() - t0, 3),
            extras={"stage": "sam3_instance", "traceback": traceback.format_exc()},
        )

    detections = list(result.detections or [])
    mask_vis: Optional[Image.Image] = None
    bbox_vis: Optional[Image.Image] = None
    if detections:
        mask_vis, bbox_vis = render_sam3_mask_bbox_previews(
            image_for_seg, detections, prompt=prompt_text
        )
    else:
        fallback = vis or decode_sam3_raw_visualization(result, image_for_seg)
        mask_vis = fallback
        bbox_vis = fallback

    id_map: Optional[np.ndarray] = None
    instance_count = 0
    if detections:
        try:
            id_map = build_instance_id_mask_from_detections(detections, (depth_w, depth_h))
            instance_count = int(id_map.max())
        except Exception as exc:
            logger.exception("build instance mask failed: %s", exc)

    # --- 2) 标记位 P1 ---
    marker_payload: Dict[str, Any] = {"enabled": bool(enable_marker_p1)}
    p1_vis: Optional[Image.Image] = None
    p1_pose: Optional[Pose6D] = None
    if enable_marker_p1:
        marker_prompt_text = (
            marker_prompt or load_marker_prompt() or DEFAULT_PLACE_MARKER_PROMPT
        ).strip()
        marker_payload["prompt"] = marker_prompt_text
        try:
            t_m = time.perf_counter()
            marker_result, _ = infer_sam3_with_image(
                image_for_seg,
                prompt=marker_prompt_text,
                api_url=sam_api,
                threshold=float(thr if thr is not None else DEFAULT_PLACE_SAM3_THRESHOLD),
                mask_threshold=float(
                    mask_thr if mask_thr is not None else DEFAULT_PLACE_SAM3_MASK_THRESHOLD
                ),
                timeout_s=timeout,
                return_vis_base64=True,
                filter_by_point=False,
            )
            marker_payload["sam3_elapsed_s"] = round(time.perf_counter() - t_m, 3)
            marker_payload["num_detections"] = marker_result.num_detections
            marker_payload["detections"] = _detection_summary(list(marker_result.detections or []))

            if not marker_result.detections:
                marker_payload["success"] = False
                marker_payload["message"] = "SAM3 未检测到标记位"
            else:
                det, selection = select_marker_detection(
                    list(marker_result.detections),
                    (depth_w, depth_h),
                    image_for_seg,
                )
                marker_payload["selection"] = selection
                if det is None:
                    marker_payload["success"] = False
                    marker_payload["message"] = "未能选出标记位实例"
                else:
                    marker_mask = _decode_detection_mask(det, (depth_w, depth_h))
                    p1_pose, p1_meta, p1_vis = estimate_p1_from_marker(
                        image_for_seg,
                        marker_mask,
                        sensor_depth,
                        place_camera,
                    )
                    marker_payload["success"] = True
                    marker_payload["p1"] = p1_pose.to_dict()
                    marker_payload["p1_meta"] = p1_meta
        except Exception as exc:
            logger.exception("marker p1 failed")
            marker_payload["success"] = False
            marker_payload["message"] = str(exc)
            marker_payload["traceback"] = traceback.format_exc()

    # --- 3) q_i ---
    if id_map is None:
        id_map = np.zeros((depth_h, depth_w), dtype=np.uint8)

    qi_radius = float(radius_mm) if radius_mm is not None else float(y_band_mm)
    qi_all = find_instance_qi_from_pi_sphere(
        sensor_depth, id_map, intrinsics, radius_mm=qi_radius
    )
    rotation_src = p1_pose.rotation if p1_pose is not None else np.eye(3, dtype=np.float64)
    if p1_pose is not None and qi_all:
        qi_all = annotate_p1_x_distance(qi_all, p1_pose.position_mm, p1_pose.rotation)
        qi_list = select_nearest_along_p1_x(qi_all)
    else:
        qi_list = list(qi_all)

    for item in qi_list:
        item["rotation_from"] = "p1" if p1_pose is not None else "identity"
        item["rotation_matrix"] = np.asarray(rotation_src, dtype=np.float64).round(6).tolist()

    # --- 4) 可视化 + 抓取 JSON ---
    base_img = mask_vis if mask_vis is not None else image_for_seg
    grasp_vis = render_p1_and_pi_preview(
        base_img,
        place_camera,
        p1=p1_pose,
        pi_list=qi_list,
        title="P1 / nearest q_i on P1-X",
        point_prefix="q",
        dist_key="x_dist_mm" if (qi_list and "x_dist_mm" in qi_list[0]) else "z_mm",
        dist_label="|dx|" if (qi_list and "x_dist_mm" in qi_list[0]) else "z",
        draw_link_to_p1=True,
    )

    if not qi_list:
        return GraspInferResult(
            success=False,
            message="未得到可用的 q 点（需实例分割成功；建议开启标记位 P1）",
            elapsed_s=round(time.perf_counter() - t0, 3),
            grasp_vis=grasp_vis,
            mask_vis=mask_vis,
            bbox_vis=bbox_vis,
            p1_vis=p1_vis,
            num_instances=instance_count,
            marker_p1=marker_payload,
            instance_qi=[],
            instance_qi_all=qi_all,
            image_size=[depth_w, depth_h],
            extras={"instance_prompt": prompt_text, "sam3_api": sam_api},
        )

    q = qi_list[0]
    grasp_pose = format_grasp_xyzrxryrz(
        q.get("q_i_mm") or q.get("position_mm"),
        rotation_src,
        meta={
            "instance_id": q.get("instance_id"),
            "p_i_mm": q.get("p_i_mm"),
            "q_i_mm": q.get("q_i_mm") or q.get("position_mm"),
            "x_dist_mm": q.get("x_dist_mm"),
            "num_sphere": q.get("num_sphere") or q.get("num_band"),
            "radius_mm": q.get("radius_mm") or q.get("y_band_mm"),
            "y_band_mm": q.get("radius_mm") or q.get("y_band_mm"),
            "rotation_from": q.get("rotation_from"),
            "role": "gripper_grasp_point",
        },
    )

    return GraspInferResult(
        success=True,
        message="ok",
        xyzrxryrz=list(grasp_pose["xyzrxryrz"]),
        grasp_pose=grasp_pose,
        grasp_vis=grasp_vis,
        mask_vis=mask_vis,
        bbox_vis=bbox_vis,
        p1_vis=p1_vis,
        elapsed_s=round(time.perf_counter() - t0, 3),
        num_instances=instance_count,
        marker_p1=marker_payload,
        instance_qi=qi_list,
        instance_qi_all=qi_all,
        image_size=[depth_w, depth_h],
        extras={
            "instance_prompt": prompt_text,
            "sam3_api": sam_api,
            "threshold": thr,
            "mask_threshold": mask_thr,
            "radius_mm": qi_radius,
            "y_band_mm": qi_radius,
            "detections": _detection_summary(detections),
        },
    )


def camera_from_dict(data: Dict[str, Any]) -> CameraIntrinsics:
    cam_k = data.get("cam_K")
    if not isinstance(cam_k, list) or len(cam_k) != 9:
        raise ValueError("camera.json cam_K 必须为 9 元素数组")
    return CameraIntrinsics(
        fx=float(cam_k[0]),
        fy=float(cam_k[4]),
        cx=float(cam_k[2]),
        cy=float(cam_k[5]),
        depth_scale=float(data.get("depth_scale", 0.001)),
    )


def infer_grasp_from_uploads(
    rgb: Image.Image,
    depth_image: Image.Image,
    camera_json: Dict[str, Any] | str | bytes,
    **kwargs: Any,
) -> GraspInferResult:
    """从上传的 RGB / 深度图 / camera.json 解析后调用 ``infer_grasp``。"""
    import json as _json

    if isinstance(camera_json, bytes):
        camera_data = _json.loads(camera_json.decode("utf-8"))
    elif isinstance(camera_json, str):
        camera_data = _json.loads(camera_json)
    else:
        camera_data = camera_json

    try:
        place_camera = camera_from_dict(camera_data)
    except Exception as exc:
        return GraspInferResult(success=False, message=f"无效 camera.json: {exc}")

    intrinsics = load_intrinsics_from_dict(camera_data)
    if intrinsics is None:
        return GraspInferResult(success=False, message="camera.json 缺少有效 cam_K")

    depth_scale = float(camera_data.get("depth_scale", 1.0))
    sensor_depth = load_sensor_depth_from_image(depth_image, depth_scale)

    return infer_grasp(
        rgb,
        sensor_depth,
        place_camera,
        intrinsics,
        **kwargs,
    )
