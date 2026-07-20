"""Gradio Tab：SAM3 文本分割 + 传感器深度点云 + 标记位 P1。"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import gradio as gr
import numpy as np
import trimesh
from PIL import Image

from src.depth.pointcloud import (
    annotate_p1_x_distance,
    camera_pose_mm_to_glb,
    create_pose_axes_mesh,
    create_pose_marker_sphere,
    depth_instance_to_pointcloud,
    export_pointcloud_ply,
    export_scene_glb,
    find_instance_nearest_p1_x_points,
    find_instance_qi_from_pi_sphere,
    inject_axis_points,
    select_nearest_along_p1_x,
)
from src.depth.service import (
    depth_to_colormap_with_colorbar,
    load_intrinsics_from_dict,
    load_sensor_depth_from_image,
)
from src.grasp.grasp_infer import format_grasp_xyzrxryrz
from src.grasp.marker import (
    camera_from_path,
    estimate_p1_from_marker,
    load_marker_prompt,
    render_p1_and_pi_preview,
    select_marker_detection,
)
from src.grasp.sam3 import (
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
    _VIS_COLORS_BGR,
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

if TYPE_CHECKING:
    from src.depth.service import DepthService

logger = logging.getLogger("sam3_tab")

Sam3TabOutputs = Tuple[
    Optional[Image.Image],  # sensor depth
    Optional[Image.Image],  # instance mask
    Optional[Image.Image],  # instance bbox
    Optional[Image.Image],  # marker p1 vis
    Optional[Image.Image],  # p1 + q_i preview
    Optional[Image.Image],  # p1 + p_ix preview
    Optional[str],  # glb preview
    Optional[str],  # glb download
    Optional[str],  # ply download
    str,  # grasp pose json (q point)
    str,  # detail json
]


def _filepath_from_upload(file_obj: Any) -> Optional[Path]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if isinstance(file_obj, dict):
        path = file_obj.get("path") or file_obj.get("name")
        if path:
            return Path(path)
    name = getattr(file_obj, "name", None)
    if name:
        return Path(name)
    return None


def _vis_colors_rgb() -> List[Tuple[int, int, int]]:
    return [(int(r), int(g), int(b)) for (b, g, r) in _VIS_COLORS_BGR]


def _empty_grasp_json(message: str = "尚未计算出抓取点 q") -> str:
    return json.dumps(
        {
            "success": False,
            "message": message,
            "xyzrxryrz": None,
            "unit": {"xyz": "mm", "rpy": "deg", "euler": "ZYX → [rx,ry,rz]=[X,Y,Z]"},
        },
        ensure_ascii=False,
        indent=2,
    )


def _error_outputs(message: str, *, sensor_vis: Optional[Image.Image] = None) -> Sam3TabOutputs:
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
    return sensor_vis, None, None, None, None, None, None, None, None, _empty_grasp_json(message), err


def _detection_summary(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, det in enumerate(detections, start=1):
        rows.append(
            {
                "id": idx,
                "score": float(det.get("score", 0.0)),
                "bbox": det.get("bbox"),
            }
        )
    return rows


def _build_scene_with_optional_p1(
    *,
    depth_service: "DepthService",
    sensor_depth: np.ndarray,
    intrinsics: np.ndarray,
    id_map: Optional[np.ndarray],
    p1_pose: Any = None,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """实例分色点云 + P1 / q_i 醒目标记（烧进点云 + mesh）→ GLB/PLY。"""
    h, w = sensor_depth.shape
    if id_map is None:
        id_map = np.zeros((h, w), dtype=np.uint8)

    points, colors, stats = depth_instance_to_pointcloud(
        depth=sensor_depth,
        instance_id_map=id_map,
        intrinsics=intrinsics,
        instance_colors_rgb=_vis_colors_rgb(),
        max_points=depth_service.max_points,
    )

    # p_i → 球半径 8mm → 相机 y±2mm 聚合得 q_i；UI / 点云展示用 q_i
    qi_all = find_instance_qi_from_pi_sphere(
        sensor_depth, id_map, intrinsics, radius_mm=8.0, y_band_mm=2.0
    )
    rotation_src = p1_pose.rotation if p1_pose is not None else np.eye(3, dtype=np.float64)

    if p1_pose is not None and qi_all:
        qi_all = annotate_p1_x_distance(qi_all, p1_pose.position_mm, p1_pose.rotation)
        qi_list = select_nearest_along_p1_x(qi_all)
    else:
        qi_list = list(qi_all)

    # Gradio Model3D 对 PointCloud 最稳：把 q_i / 轴方向直接画成点
    marker_pts: list[np.ndarray] = []
    marker_cols: list[np.ndarray] = []
    geometries: list = []

    pi_core_colors = (
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 128, 0),
        (0, 255, 128),
        (255, 64, 128),
    )

    if p1_pose is not None:
        origin_p1, rot_p1 = camera_pose_mm_to_glb(p1_pose.position_mm, p1_pose.rotation)
        p_pts, p_cols = inject_axis_points(
            origin_p1, rot_p1, axis_length_mm=40.0, core_color=(255, 220, 0)
        )
        marker_pts.append(p_pts)
        marker_cols.append(p_cols)
        geometries.append(
            create_pose_marker_sphere(origin_p1, radius_mm=8.0, color_rgba=(255, 220, 0, 255))
        )
        geometries.append(
            create_pose_axes_mesh(origin_p1, rot_p1, axis_length_mm=45.0, radius_mm=1.5)
        )

    for item in qi_list:
        origin_i, rot_i = camera_pose_mm_to_glb(item["position_mm"], rotation_src)
        core_rgb = pi_core_colors[(int(item["instance_id"]) - 1) % len(pi_core_colors)]
        p_pts, p_cols = inject_axis_points(
            origin_i, rot_i, axis_length_mm=55.0, core_color=core_rgb
        )
        marker_pts.append(p_pts)
        marker_cols.append(p_cols)
        geometries.append(
            create_pose_marker_sphere(origin_i, radius_mm=12.0, color_rgba=(*core_rgb, 255))
        )
        geometries.append(
            create_pose_axes_mesh(origin_i, rot_i, axis_length_mm=70.0, radius_mm=2.2)
        )
        item["rotation_from"] = "p1" if p1_pose is not None else "identity"
        item["rotation_matrix"] = np.asarray(rotation_src, dtype=np.float64).round(6).tolist()
        item["glb_position_mm"] = [
            round(float(origin_i[0]), 2),
            round(float(origin_i[1]), 2),
            round(float(origin_i[2]), 2),
        ]

    if marker_pts:
        points = (
            np.vstack([points, *marker_pts])
            if points.size
            else np.vstack(marker_pts)
        )
        colors = (
            np.vstack([colors, *marker_cols])
            if colors.size
            else np.vstack(marker_cols)
        )

    stem = f"sam3_{uuid.uuid4().hex[:10]}"
    ply_path = depth_service.output_dir / f"{stem}_instance.ply"
    glb_path = depth_service.output_dir / f"{stem}_instance.glb"
    export_pointcloud_ply(points, colors, ply_path)

    if points.size == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=[[128, 128, 128]])
    else:
        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    geometries = [cloud, *geometries]

    export_scene_glb(geometries, glb_path)
    logger.info(
        "scene built points=%s instances=%s qi=%s p1=%s",
        points.shape[0],
        stats.get("instance_ids"),
        len(qi_list),
        p1_pose is not None,
    )
    return str(glb_path), str(ply_path), {
        "point_count": int(points.shape[0]),
        "point_stats": stats,
        "p1_axes": p1_pose is not None,
        "instance_qi_all": qi_all,
        "instance_qi": qi_list,
        # 兼容旧字段名：UI 原先读 instance_pi，现为 q_i
        "instance_pi_all": qi_all,
        "instance_pi": qi_list,
        "pi_count": len(qi_list),
        "pi_candidate_count": len(qi_all),
        "pi_note": (
            "各实例：p_i=剔除外点后 Z 最小 → 球半径 8mm → 相机 y±2mm 聚合中心 q_i；"
            "有 P1 时再按 q_i 的 P1-X |dx| 只保留最近 1 个用于显示；"
            "JSON 含 p_i_mm / q_i_mm，候选见 instance_qi_all"
        ),
    }


def run_sam3_seg_tab_inference(
    depth_service: "DepthService",
    image: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    prompt: str,
    marker_prompt: str,
    enable_marker_p1: bool,
    api_url: str,
    threshold: float,
    mask_threshold: float,
    timeout_s: float,
) -> Sam3TabOutputs:
    if image is None:
        return _error_outputs("请先上传 RGB 图片")

    depth_path = _filepath_from_upload(depth_file)
    camera_path = _filepath_from_upload(camera_file)
    if depth_path is None or not depth_path.is_file():
        return _error_outputs("请上传传感器深度图（uint16 .png）")
    if camera_path is None or not camera_path.is_file():
        return _error_outputs("请上传 camera.json（点云反投影需要内参）")

    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return _error_outputs("SAM3 实例提示词不能为空")

    try:
        with camera_path.open(encoding="utf-8") as f:
            intrinsics_data = json.load(f)
        intrinsics = load_intrinsics_from_dict(intrinsics_data)
        if intrinsics is None:
            return _error_outputs("camera.json 缺少有效 cam_K")
        depth_scale = float(intrinsics_data.get("depth_scale", 1.0))
        sensor_depth = load_sensor_depth_from_image(Image.open(depth_path), depth_scale)
        place_camera = camera_from_path(camera_path)
    except Exception as exc:
        return _error_outputs(f"读取深度/内参失败: {exc}")

    sensor_vis = depth_to_colormap_with_colorbar(sensor_depth, unit="mm")

    depth_h, depth_w = sensor_depth.shape
    rgb_w, rgb_h = image.size
    if (rgb_w, rgb_h) != (depth_w, depth_h):
        image_for_seg = image.convert("RGB").resize((depth_w, depth_h), Image.BILINEAR)
        logger.warning(
            "RGB %sx%s ≠ depth %sx%s，已将 RGB resize 到 depth 尺寸",
            rgb_w,
            rgb_h,
            depth_w,
            depth_h,
        )
    else:
        image_for_seg = image.convert("RGB")

    sam_api = (api_url or "").strip() or DEFAULT_SAM3_API_URL
    t0 = time.perf_counter()

    # --- 1) 实例分割（用户提示词）---
    try:
        result, vis = infer_sam3_with_image(
            image_for_seg,
            prompt=prompt_text,
            api_url=sam_api,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            timeout_s=float(timeout_s),
            return_vis_base64=True,
            filter_by_point=False,
        )
    except Exception as exc:
        err = {
            "success": False,
            "stage": "sam3_instance",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return (
            sensor_vis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            _empty_grasp_json(str(exc)),
            json.dumps(err, ensure_ascii=False, indent=2),
        )

    detections = list(result.detections or [])
    mask_vis: Optional[Image.Image] = None
    bbox_vis: Optional[Image.Image] = None
    if detections:
        mask_vis, bbox_vis = render_sam3_mask_bbox_previews(
            image_for_seg, detections, prompt=prompt_text
        )
    else:
        # 无 detection 时退回服务端整图（可能含 mask+bbox）
        fallback = vis
        if fallback is None:
            fallback = decode_sam3_raw_visualization(result, image_for_seg)
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

    # --- 2) 标记位 → P1（可选，独立 SAM3 提示词）---
    marker_payload: Dict[str, Any] = {"enabled": bool(enable_marker_p1)}
    p1_vis: Optional[Image.Image] = None
    p1_pose = None
    if enable_marker_p1:
        marker_prompt_text = (marker_prompt or load_marker_prompt() or DEFAULT_PLACE_MARKER_PROMPT).strip()
        marker_payload["prompt"] = marker_prompt_text
        try:
            t_m = time.perf_counter()
            marker_result, _ = infer_sam3_with_image(
                image_for_seg,
                prompt=marker_prompt_text,
                api_url=sam_api,
                threshold=float(threshold if threshold is not None else DEFAULT_PLACE_SAM3_THRESHOLD),
                mask_threshold=float(
                    mask_threshold if mask_threshold is not None else DEFAULT_PLACE_SAM3_MASK_THRESHOLD
                ),
                timeout_s=float(timeout_s),
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
            logger.exception("marker p1 failed: %s", exc)
            marker_payload["success"] = False
            marker_payload["message"] = str(exc)
            marker_payload["traceback"] = traceback.format_exc()

    # --- 3) 点云 ---
    glb_path: Optional[str] = None
    ply_path: Optional[str] = None
    pc_stats: Dict[str, Any] = {}
    try:
        glb_path, ply_path, pc_stats = _build_scene_with_optional_p1(
            depth_service=depth_service,
            sensor_depth=sensor_depth,
            intrinsics=intrinsics,
            id_map=id_map,
            p1_pose=p1_pose,
        )
    except Exception as exc:
        logger.exception("pointcloud failed: %s", exc)
        pc_stats = {"error": str(exc)}

    # --- 4) 2D 预览：P1 + 最近（沿 P1-X）的 q_i / p_ix ---
    pi_preview: Optional[Image.Image] = None
    pix_preview: Optional[Image.Image] = None
    pix_list: list = []
    try:
        base_img = mask_vis if mask_vis is not None else image_for_seg
        qi_selected = list(pc_stats.get("instance_qi") or pc_stats.get("instance_pi") or [])
        pi_preview = render_p1_and_pi_preview(
            base_img,
            place_camera,
            p1=p1_pose,
            pi_list=qi_selected,
            title="P1 / nearest q_i on P1-X",
            point_prefix="q",
            dist_key="x_dist_mm" if (qi_selected and "x_dist_mm" in qi_selected[0]) else "z_mm",
            dist_label="|dx|" if (qi_selected and "x_dist_mm" in qi_selected[0]) else "z",
            draw_link_to_p1=True,
        )
        if p1_pose is not None and id_map is not None:
            pix_all = find_instance_nearest_p1_x_points(
                sensor_depth,
                id_map,
                intrinsics,
                p1_pose.position_mm,
                p1_pose.rotation,
            )
            for item in pix_all:
                item["rotation_from"] = "p1"
                item["rotation_matrix"] = np.asarray(p1_pose.rotation, dtype=np.float64).round(6).tolist()
            pix_list = select_nearest_along_p1_x(pix_all)
            pc_stats["instance_pi_x_all"] = pix_all
            pc_stats["instance_pi_x"] = pix_list
            pc_stats["pi_x_count"] = len(pix_list)
            pc_stats["pi_x_candidate_count"] = len(pix_all)
            pc_stats["pi_x_note"] = (
                "各实例先取沿 P1-X 最近的 p_ix；再对全部实例按 |dx| 排序，"
                "只保留最近的 1 个显示；候选见 instance_pi_x_all"
            )
            pix_preview = render_p1_and_pi_preview(
                base_img,
                place_camera,
                p1=p1_pose,
                pi_list=pix_list,
                title="P1 / nearest p_ix on P1-X",
                point_prefix="px",
                dist_key="x_dist_mm",
                dist_label="|dx|",
                draw_link_to_p1=True,
            )
        else:
            pc_stats["instance_pi_x_all"] = []
            pc_stats["instance_pi_x"] = []
            pc_stats["pi_x_count"] = 0
            pc_stats["pi_x_note"] = "需要成功计算 P1 才能求沿 X 最近的 p_ix"
    except Exception as exc:
        logger.exception("p1/pi preview failed: %s", exc)

    elapsed_s = time.perf_counter() - t0
    raw = dict(result.raw_response or {})
    raw.pop("visualization_base64", None)

    qi_selected = list(pc_stats.get("instance_qi") or pc_stats.get("instance_pi") or [])
    grasp_payload: Dict[str, Any]
    if qi_selected:
        q = qi_selected[0]
        rot = q.get("rotation_matrix")
        if rot is None and p1_pose is not None:
            rot = p1_pose.rotation
        if rot is None:
            rot = np.eye(3, dtype=np.float64)
        grasp_payload = format_grasp_xyzrxryrz(
            q.get("q_i_mm") or q.get("position_mm"),
            rot,
            meta={
                "instance_id": q.get("instance_id"),
                "p_i_mm": q.get("p_i_mm"),
                "q_i_mm": q.get("q_i_mm") or q.get("position_mm"),
                "x_dist_mm": q.get("x_dist_mm"),
                "num_sphere": q.get("num_sphere"),
                "num_band": q.get("num_band"),
                "radius_mm": q.get("radius_mm"),
                "y_band_mm": q.get("y_band_mm"),
                "rotation_from": q.get("rotation_from", "p1" if p1_pose is not None else "identity"),
                "role": "gripper_grasp_point",
            },
        )
    else:
        grasp_payload = json.loads(_empty_grasp_json("无可用的 q 点（需实例分割成功，建议开启标记位 P1）"))

    grasp_json = json.dumps(grasp_payload, ensure_ascii=False, indent=2)

    payload = {
        "success": True,
        "elapsed_s": round(elapsed_s, 3),
        "num_instances": instance_count or len(detections),
        "detections": _detection_summary(detections),
        "grasp_pose": grasp_payload,
        "pointcloud": pc_stats,
        "scene_glb": glb_path,
        "sensor_valid_ratio": float(
            (np.isfinite(sensor_depth) & (sensor_depth > 0)).mean()
        ),
        "image_size": [depth_w, depth_h],
        "sam3_api": sam_api,
        "instance_prompt": prompt_text,
        "threshold": float(threshold),
        "mask_threshold": float(mask_threshold),
        "marker_p1": marker_payload,
        "raw_response": raw,
    }
    return (
        sensor_vis,
        mask_vis,
        bbox_vis,
        p1_vis,
        pi_preview,
        pix_preview,
        glb_path,
        glb_path,
        ply_path,
        grasp_json,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def build_sam3_seg_tab(depth_service: "DepthService") -> None:
    """在 ``with gr.Tabs():`` 内调用，添加「抓取 位姿估计」页签。"""
    default_marker = load_marker_prompt() or DEFAULT_PLACE_MARKER_PROMPT

    with gr.Tab("抓取 位姿估计"):
        gr.Markdown(
            "验证 **SAM3 文本分割** + **传感器深度点云**；可选再识别 **绿色标记位**，"
            "用 PEM 同款方法计算并展示 **P1**（对角线中心反投影 + 平面姿态）。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                sam3_rgb = gr.Image(type="pil", label="上传 RGB", height=280, interactive=True)
                with gr.Row():
                    sam3_depth = gr.File(
                        label="传感器深度（uint16 .png）",
                        type="filepath",
                        file_types=[".png", ".exr"],
                    )
                    sam3_camera = gr.File(
                        label="相机 camera.json",
                        type="filepath",
                        file_types=[".json"],
                    )
                sam3_prompt = gr.Textbox(
                    label="实例分割提示词（料盘/圆盘等）",
                    value=DEFAULT_SAM3_PROMPT,
                    lines=3,
                )
                enable_marker = gr.Checkbox(
                    label="识别标记位并计算 P1",
                    value=True,
                )
                marker_prompt = gr.Textbox(
                    label="标记位提示词 marker_prompt",
                    value=default_marker,
                    lines=2,
                )
                with gr.Accordion("SAM3 推理参数", open=True):
                    sam3_api = gr.Textbox(label="SAM3 API URL", value=DEFAULT_SAM3_API_URL)
                    sam3_threshold = gr.Slider(
                        label="Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=DEFAULT_SAM3_THRESHOLD,
                    )
                    sam3_mask_threshold = gr.Slider(
                        label="Mask Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=DEFAULT_SAM3_MASK_THRESHOLD,
                    )
                    sam3_timeout = gr.Number(
                        label="超时（秒）",
                        value=DEFAULT_SAM3_TIMEOUT_S,
                        precision=0,
                    )
                sam3_btn = gr.Button("开始抓取位姿估计", variant="primary")
                out_grasp = gr.Code(
                    label="抓取点 q · xyzrxryrz（mm / °）",
                    language="json",
                    lines=12,
                    value=_empty_grasp_json(),
                )

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                out_sensor = gr.Image(type="pil", label="传感器深度（伪彩色）", height=200)
                out_seg_mask = gr.Image(type="pil", label="SAM3 实例 mask", height=200)
                out_seg_bbox = gr.Image(type="pil", label="SAM3 实例 bbox", height=200)
                out_p1 = gr.Image(
                    type="pil",
                    label="标记位 P1（mask / 角点 / 坐标轴）",
                    height=220,
                )
                out_pi = gr.Image(
                    type="pil",
                    label="P1 + 最近 q_i（球 8mm + y±2mm；按 P1-X 仅 1 个）",
                    height=240,
                )
                out_pix = gr.Image(
                    type="pil",
                    label="P1 + 最近 p_ix（按 P1-X |dx|，仅 1 个）",
                    height=240,
                )

        gr.Markdown("### 3D 点云（传感器深度 · 实例分色 · P1 / q_i 坐标轴）")
        gr.Markdown(
            "> **灰色**=背景；**彩色**=各 SAM3 实例；"
            "每实例先求 **p_i（Z 最小）**，再 **球半径 8mm** 后 **相机 y±2mm** 聚合得 **q_i** 用于展示；"
            "有 P1 时再按 q_i 的 P1-X |dx| **只保留最近 1 个**。"
            "黄球/短轴 = 标记 P1。详情见 JSON `instance_qi`（含 `p_i_mm` / `q_i_mm`）。"
        )
        with gr.Row():
            with gr.Column(scale=4):
                out_glb = gr.Model3D(label="浏览器预览（GLB）", height=520)
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_glb_dl = gr.File(label="GLB", interactive=False)
                out_ply_dl = gr.File(label="PLY", interactive=False)

        out_json = gr.Code(label="分割 / P1 / 点云详情", language="json", lines=16)

        def _run(*args):
            return run_sam3_seg_tab_inference(depth_service, *args)

        sam3_btn.click(
            fn=_run,
            inputs=[
                sam3_rgb,
                sam3_depth,
                sam3_camera,
                sam3_prompt,
                marker_prompt,
                enable_marker,
                sam3_api,
                sam3_threshold,
                sam3_mask_threshold,
                sam3_timeout,
            ],
            outputs=[
                out_sensor,
                out_seg_mask,
                out_seg_bbox,
                out_p1,
                out_pi,
                out_pix,
                out_glb,
                out_glb_dl,
                out_ply_dl,
                out_grasp,
                out_json,
            ],
        )

        gr.Markdown(
            f"""
            **说明**
            - 实例分割：SAM3 `POST /infer` + 文本提示，默认 API `{DEFAULT_SAM3_API_URL}`
              （快速预览拆成 **mask** / **bbox** 两张图）
            - **标记位 P1**（与 PEM_service 一致）：
              1. 用 marker 提示词分割 → 按绿度/长宽比选出标记
              2. 四角 A–D 对角线交点为中心 UV，邻域深度反投影得位置
              3. 辅助点 A'–D' 拟合平面法向 → X/Y/Z 姿态
            - **抓取点 q**：最终保留的 q_i；左侧 JSON 的 `xyzrxryrz = [x,y,z,rx,ry,rz]`（xyz=mm，姿态=°，ZYX）
            - **实例 q_i**：先求 **p_i（Z 最小）** → **球 8mm** → **相机 y±2mm** 聚合中心 **q_i**（UI 展示 q_i）
              → 有 P1 时再按 **P1-X |dx|** **只显示最近的 1 个**
            - **实例 p_ix**：各实例沿 P1-X 最近点后，再按 **|dx|** 排序，**只显示最近的 1 个**
            - **快速预览**：q_i / p_ix 图均只画选出的那一个点；候选在 JSON `instance_qi_all` / `instance_pi_x_all`
            - 点云仅用 **原始传感器深度**；3D 标记为 **q_i**（彩色密集球 + RGB 三色射线）
            - 黄球/短轴 = 标记 **P1**；请同时核对 JSON 的 `p_i_mm` / `q_i_mm` / `instance_qi`
            - 默认标记提示词：`{default_marker}`
            """
        )
