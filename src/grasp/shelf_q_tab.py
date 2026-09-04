"""Gradio Tab：P1 按上下孔线沿高度上移得到抓取点 Q。"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import gradio as gr
import numpy as np
import trimesh
from PIL import Image

from src.depth.pointcloud import (
    camera_pose_mm_to_glb,
    create_pose_axes_mesh,
    create_pose_marker_sphere,
    depth_instance_to_pointcloud,
    export_pointcloud_ply,
    export_scene_glb,
    inject_axis_points,
)
from src.depth.service import (
    depth_to_colormap_with_colorbar,
    load_intrinsics_from_dict,
    load_sensor_depth_from_image,
)
from src.grasp.grasp_infer import format_grasp_xyzrxryrz
from src.grasp.marker import (
    camera_from_path,
    infer_p1_from_shelf_panel,
    load_hole_prompt,
    load_led_prompt,
    render_p1_q_offset_preview,
)
from src.grasp.place_geometry import Pose6D, compute_q_from_p1_hole_row
from src.grasp.sam3 import (
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
    build_instance_id_mask_from_detections,
    decode_sam3_raw_visualization,
    infer_sam3_with_image,
    render_sam3_mask_bbox_previews,
)
from src.grasp.sam3_tab import (
    _detection_summary,
    _filepath_from_upload,
    _vis_colors_rgb,
)
from src.grasp.settings import (
    DEFAULT_GRASP_LED_BOTTOM_UP_MM,
    DEFAULT_GRASP_LED_TOP_UP_MM,
    DEFAULT_PLACE_HOLE_PROMPT,
    DEFAULT_PLACE_LED_PROMPT,
    DEFAULT_PLACE_SAM3_API_URL,
    DEFAULT_PLACE_SAM3_MASK_THRESHOLD,
    DEFAULT_PLACE_SAM3_THRESHOLD,
)

if TYPE_CHECKING:
    from src.depth.service import DepthService

logger = logging.getLogger("shelf_q_tab")

ShelfQTabOutputs = Tuple[
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[str],
    Optional[str],
    Optional[str],
    str,
    str,
]


def _empty_q_json(message: str = "尚未计算出抓取点 Q") -> str:
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


def _error_outputs(message: str, *, sensor_vis: Optional[Image.Image] = None) -> ShelfQTabOutputs:
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
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
        _empty_q_json(message),
        err,
    )


def _build_scene_p1_q(
    *,
    depth_service: "DepthService",
    sensor_depth: np.ndarray,
    intrinsics: np.ndarray,
    id_map: Optional[np.ndarray],
    p1_pose: Optional[Pose6D],
    q_pose: Optional[Pose6D],
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
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

    marker_pts: list[np.ndarray] = []
    marker_cols: list[np.ndarray] = []
    geometries: list = []

    if p1_pose is not None:
        origin_p1, rot_p1 = camera_pose_mm_to_glb(p1_pose.position_mm, p1_pose.rotation)
        p_pts, p_cols = inject_axis_points(
            origin_p1,
            rot_p1,
            axis_length_mm=(120.0, 40.0, 40.0),
            core_color=(255, 220, 0),
        )
        marker_pts.append(p_pts)
        marker_cols.append(p_cols)
        geometries.append(
            create_pose_marker_sphere(origin_p1, radius_mm=8.0, color_rgba=(255, 220, 0, 255))
        )
        geometries.append(
            create_pose_axes_mesh(
                origin_p1, rot_p1, axis_length_mm=(135.0, 45.0, 45.0), radius_mm=1.5
            )
        )

    if q_pose is not None:
        origin_q, rot_q = camera_pose_mm_to_glb(q_pose.position_mm, q_pose.rotation)
        p_pts, p_cols = inject_axis_points(
            origin_q, rot_q, axis_length_mm=70.0, core_color=(255, 0, 255)
        )
        marker_pts.append(p_pts)
        marker_cols.append(p_cols)
        geometries.append(
            create_pose_marker_sphere(origin_q, radius_mm=5.0, color_rgba=(255, 0, 255, 255))
        )
        geometries.append(
            create_pose_axes_mesh(origin_q, rot_q, axis_length_mm=80.0, radius_mm=2.2)
        )

    if marker_pts:
        points = np.vstack([points, *marker_pts]) if points.size else np.vstack(marker_pts)
        colors = np.vstack([colors, *marker_cols]) if colors.size else np.vstack(marker_cols)

    stem = f"shelf_q_{uuid.uuid4().hex[:10]}"
    ply_path = depth_service.output_dir / f"{stem}_instance.ply"
    glb_path = depth_service.output_dir / f"{stem}_instance.glb"
    export_pointcloud_ply(points, colors, ply_path)

    if points.size == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=[[128, 128, 128]])
    else:
        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    export_scene_glb([cloud, *geometries], glb_path)
    return str(glb_path), str(ply_path), {
        "point_count": int(points.shape[0]),
        "point_stats": stats,
        "p1_axes": p1_pose is not None,
        "q_axes": q_pose is not None,
        "pi_note": (
            "黄球 = P1；品红球 = Q（LED 在上排孔线上移 24.5mm / 下排上移 30mm）；"
            "数值 frame=camera；3D 预览 preview_frame=glb_y_up（Y 翻转）"
        ),
    }


def run_shelf_q_tab_inference(
    depth_service: "DepthService",
    image: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    prompt: str,
    hole_prompt: str,
    led_prompt: str,
    enable_marker_p1: bool,
    api_url: str,
    marker_api_url: str,
    threshold: float,
    marker_threshold: float,
    mask_threshold: float,
    timeout_s: float,
) -> ShelfQTabOutputs:
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
    else:
        image_for_seg = image.convert("RGB")

    sam_api = (api_url or "").strip() or DEFAULT_SAM3_API_URL
    marker_api = (marker_api_url or "").strip() or DEFAULT_PLACE_SAM3_API_URL
    t0 = time.perf_counter()

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
            _empty_q_json(str(exc)),
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

    marker_payload: Dict[str, Any] = {"enabled": bool(enable_marker_p1)}
    p1_vis: Optional[Image.Image] = None
    abcd_zoom: Optional[Image.Image] = None
    p1_pose = None
    if enable_marker_p1:
        hole_prompt_text = (hole_prompt or load_hole_prompt() or DEFAULT_PLACE_HOLE_PROMPT).strip()
        led_prompt_text = (led_prompt or load_led_prompt() or DEFAULT_PLACE_LED_PROMPT).strip()
        marker_payload["hole_prompt"] = hole_prompt_text
        marker_payload["led_prompt"] = led_prompt_text
        marker_payload["sam3_api"] = marker_api
        try:
            p1_pose, marker_payload, p1_vis, abcd_zoom = infer_p1_from_shelf_panel(
                image_for_seg,
                sensor_depth,
                place_camera,
                (depth_w, depth_h),
                hole_prompt=hole_prompt_text,
                led_prompt=led_prompt_text,
                api_url=marker_api,
                threshold=float(
                    marker_threshold
                    if marker_threshold is not None
                    else DEFAULT_PLACE_SAM3_THRESHOLD
                ),
                mask_threshold=float(
                    mask_threshold if mask_threshold is not None else DEFAULT_PLACE_SAM3_MASK_THRESHOLD
                ),
                timeout_s=float(timeout_s),
            )
            if p1_pose is None and "message" not in marker_payload:
                marker_payload["message"] = "P1 计算失败"
        except Exception as exc:
            logger.exception("marker p1 failed: %s", exc)
            marker_payload["success"] = False
            marker_payload["message"] = str(exc)
            marker_payload["traceback"] = traceback.format_exc()

    q_pose: Optional[Pose6D] = None
    q_meta: Dict[str, Any] = {}
    if p1_pose is not None:
        p1_meta = marker_payload.get("p1_meta") or {}
        hole_line = p1_meta.get("hole_line") or {}
        q_pose, q_meta = compute_q_from_p1_hole_row(
            p1_pose,
            hole_line,
            top_up_mm=float(DEFAULT_GRASP_LED_TOP_UP_MM),
            bottom_up_mm=float(DEFAULT_GRASP_LED_BOTTOM_UP_MM),
        )

    q_preview = render_p1_q_offset_preview(
        mask_vis if mask_vis is not None else image_for_seg,
        place_camera,
        p1=p1_pose,
        q=q_pose,
        hole_line=(marker_payload.get("p1_meta") or {}).get("hole_line"),
        row_label=str(q_meta.get("row") or ""),
        offset_mm=float(q_meta.get("up_mm") or 0.0),
    )

    glb_path: Optional[str] = None
    ply_path: Optional[str] = None
    pc_stats: Dict[str, Any] = {}
    try:
        glb_path, ply_path, pc_stats = _build_scene_p1_q(
            depth_service=depth_service,
            sensor_depth=sensor_depth,
            intrinsics=intrinsics,
            id_map=id_map,
            p1_pose=p1_pose,
            q_pose=q_pose,
        )
    except Exception as exc:
        logger.exception("pointcloud failed: %s", exc)
        pc_stats = {"error": str(exc)}

    if q_pose is not None:
        grasp_payload = format_grasp_xyzrxryrz(
            q_pose.position_mm,
            q_pose.rotation,
            meta={
                "role": "gripper_grasp_point",
                "from": "p1_up_along_hole_row",
                **q_meta,
            },
        )
    else:
        grasp_payload = json.loads(_empty_q_json("无 P1，无法计算 Q（需孔洞 + LED）"))

    elapsed_s = time.perf_counter() - t0
    payload = {
        "success": bool(q_pose is not None),
        "elapsed_s": round(elapsed_s, 3),
        "num_instances": instance_count or len(detections),
        "detections": _detection_summary(detections),
        "grasp_pose": grasp_payload,
        "q_offset": q_meta,
        "pointcloud": pc_stats,
        "scene_glb": glb_path,
        "image_size": [depth_w, depth_h],
        "sam3_api": sam_api,
        "sam3_marker_api": marker_api,
        "instance_prompt": prompt_text,
        "marker_p1": marker_payload,
    }
    return (
        sensor_vis,
        mask_vis,
        bbox_vis,
        p1_vis,
        abcd_zoom,
        q_preview,
        glb_path,
        glb_path,
        ply_path,
        json.dumps(grasp_payload, ensure_ascii=False, indent=2),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def build_shelf_q_tab(depth_service: "DepthService") -> None:
    """在 ``with gr.Tabs():`` 内调用，添加「抓取 孔线 Q」页签。"""
    default_hole = load_hole_prompt() or DEFAULT_PLACE_HOLE_PROMPT
    default_led = load_led_prompt() or DEFAULT_PLACE_LED_PROMPT
    top_mm = float(DEFAULT_GRASP_LED_TOP_UP_MM)
    bot_mm = float(DEFAULT_GRASP_LED_BOTTOM_UP_MM)

    with gr.Tab("抓取 孔线 Q"):
        gr.Markdown(
            "与 **抓取 位姿估计** 同款 P1（孔洞双平行线 + 蓝色 LED）。"
            f"LED 在 **上排孔线** 则 P1 沿局部 +Y 上移 **{top_mm:g} mm**；"
            f"在 **下排孔线** 则上移 **{bot_mm:g} mm**，得到抓取点 **Q**。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                rgb = gr.Image(type="pil", label="上传 RGB", height=280, interactive=True)
                with gr.Row():
                    depth = gr.File(
                        label="传感器深度（uint16 .png）",
                        type="filepath",
                        file_types=[".png", ".exr"],
                    )
                    camera = gr.File(
                        label="相机 camera.json",
                        type="filepath",
                        file_types=[".json"],
                    )
                prompt = gr.Textbox(
                    label="实例分割提示词（料盘/圆盘等，仅点云分色）",
                    value=DEFAULT_SAM3_PROMPT,
                    lines=3,
                )
                enable_marker = gr.Checkbox(label="识别孔洞 + 蓝色 LED 并计算 P1 / Q", value=True)
                hole_prompt = gr.Textbox(
                    label="货架孔洞提示词 hole_prompt",
                    value=default_hole,
                    lines=2,
                )
                led_prompt = gr.Textbox(
                    label="蓝色 LED 提示词 led_prompt",
                    value=default_led,
                    lines=2,
                )
                with gr.Accordion("SAM3 推理参数", open=True):
                    sam3_api = gr.Textbox(label="料盘 SAM3 API（微调）", value=DEFAULT_SAM3_API_URL)
                    sam3_marker_api = gr.Textbox(
                        label="孔洞 / LED SAM3 API（官方）",
                        value=DEFAULT_PLACE_SAM3_API_URL,
                    )
                    sam3_threshold = gr.Slider(
                        label="Threshold（料盘 / 微调）",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=DEFAULT_SAM3_THRESHOLD,
                    )
                    sam3_marker_threshold = gr.Slider(
                        label="Threshold（孔洞 / LED / 官方）",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=DEFAULT_PLACE_SAM3_THRESHOLD,
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
                btn = gr.Button("开始计算 Q", variant="primary")
                out_grasp = gr.Code(
                    label="抓取点 Q · xyzrxryrz（mm / °）",
                    language="json",
                    lines=12,
                    value=_empty_q_json(),
                )

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                out_sensor = gr.Image(type="pil", label="传感器深度（伪彩色）", height=200)
                out_seg_mask = gr.Image(type="pil", label="SAM3 实例 mask", height=200)
                out_seg_bbox = gr.Image(type="pil", label="SAM3 实例 bbox", height=200)
                out_p1 = gr.Image(
                    type="pil",
                    label="P1（孔洞水平 + 蓝色 LED + 外接正方形 ABCD）",
                    height=220,
                )
                out_abcd = gr.Image(
                    type="pil",
                    label="ABCD+abcd 外接正方形放大（8 角点 / 深度）",
                    height=240,
                )
                out_q = gr.Image(
                    type="pil",
                    label=f"P1 → Q（上排 +{top_mm:g}mm / 下排 +{bot_mm:g}mm）",
                    height=260,
                )

        gr.Markdown("### 3D 点云（传感器深度 · P1 黄 / Q 品红）")
        with gr.Row():
            with gr.Column(scale=4):
                out_glb = gr.Model3D(label="浏览器预览（GLB）", height=520)
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_glb_dl = gr.File(label="GLB", interactive=False)
                out_ply_dl = gr.File(label="PLY", interactive=False)

        out_json = gr.Code(label="分割 / P1 / Q 详情", language="json", lines=16)

        def _run(*args):
            return run_shelf_q_tab_inference(depth_service, *args)

        btn.click(
            fn=_run,
            inputs=[
                rgb,
                depth,
                camera,
                prompt,
                hole_prompt,
                led_prompt,
                enable_marker,
                sam3_api,
                sam3_marker_api,
                sam3_threshold,
                sam3_marker_threshold,
                sam3_mask_threshold,
                sam3_timeout,
            ],
            outputs=[
                out_sensor,
                out_seg_mask,
                out_seg_bbox,
                out_p1,
                out_abcd,
                out_q,
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
            - P1 算法与「抓取 位姿估计」相同：官方 SAM3 孔洞双平行线定 X，LED 定位置
            - **Q**：LED 更靠近 `holes H/top` → 沿 P1 局部 **+Y（向上）** 平移 **{top_mm:g} mm**；
              更靠近 `holes H/bottom` → 平移 **{bot_mm:g} mm**
            - 姿态与 P1 相同；点云黄球 = P1，品红球 = Q
            """
        )
