"""融合深度 + SAM-6D 位姿 Gradio Tab 与推理逻辑。"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import gradio as gr
import numpy as np
from PIL import Image

from src.depth.pointcloud import build_pointcloud_scene_files
from src.depth.service import (
    depth_to_colormap_with_colorbar,
    load_intrinsics_from_dict,
    load_sensor_depth_from_image,
)
from src.storage.upload_archive import save_depth_mm_png
from src.grasp.paths import PROMPT_DIR
from src.grasp.pem import (
    build_sam6d_asset_urls,
    enrich_sam6d_body,
    extract_best_segmentation_mask,
    extract_pem_poses,
    fetch_sam6d_health,
    format_pem_summary,
    get_pem_sphere_radius_mm,
    image_size_wh,
    infer_sam6d,
    try_load_pem_depth_colormap,
    try_load_sam6d_visualization,
    write_camera_json_for_fused_depth,
)
from src.grasp.settings import (
    DEFAULT_SAM6D_API_URL,
    DEFAULT_SAM6D_CAD_PATH,
    DEFAULT_SAM6D_DET_SCORE_THRESH,
    DEFAULT_SAM6D_DEPTH_SOURCE,
    DEFAULT_SAM6D_FILE_SERVER_URL,
    DEFAULT_SAM6D_OUTPUT_ROOT,
    DEFAULT_SAM6D_SAM_FILE_SERVER_URL,
    DEFAULT_SAM6D_SAM_OUTPUT_ROOT,
    DEFAULT_SAM6D_SEG_BACKEND,
    DEFAULT_SAM6D_TIMEOUT_S,
    DEFAULT_VLM_API_URL,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TEMPERATURE,
    DEFAULT_VLM_TIMEOUT_S,
)
from src.grasp.sam3 import (
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
    decode_sam3_raw_visualization,
    infer_sam3_with_image,
    render_sam3_mask_bbox_previews,
)
from src.grasp.vlm import (
    VlmPointItem,
    detect_vlm_points,
    render_vlm_points_on_image,
)

if TYPE_CHECKING:
    from src.depth.service import DepthService

Sam6dUiOutputs = Tuple[
    Optional[Image.Image],  # ism mask
    Optional[Image.Image],  # ism bbox
    Optional[Image.Image],  # pem vis
    Optional[Image.Image],  # depth input vis
    str,  # json
    Optional[Image.Image],  # depth colormap
    Optional[str],  # glb preview
    Optional[str],  # glb download
    Optional[str],  # ply download
]


def load_prompt_file(filename: str) -> str:
    path = PROMPT_DIR / filename
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


DEFAULT_POINT_PROMPT = load_prompt_file("point_tray_over_marker.txt")


def _filepath_from_upload(file_obj: Any) -> Optional[Path]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str) and file_obj.strip():
        return Path(file_obj)
    if isinstance(file_obj, dict):
        path = file_obj.get("path") or file_obj.get("name")
        if path:
            return Path(path)
    name = getattr(file_obj, "name", None)
    if name:
        return Path(name)
    return None


def _sam6d_error_outputs(
    message: str,
    *,
    ism_mask_vis: Optional[Image.Image] = None,
    ism_bbox_vis: Optional[Image.Image] = None,
    pem_vis: Optional[Image.Image] = None,
    fused_depth_vis: Optional[Image.Image] = None,
    scene_glb: Optional[str] = None,
    scene_ply: Optional[str] = None,
) -> Sam6dUiOutputs:
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
    return (
        ism_mask_vis,
        ism_bbox_vis,
        pem_vis,
        fused_depth_vis,
        err,
        None,
        scene_glb,
        scene_glb,
        scene_ply,
    )


def _normalize_ism_detections(detection_ism: Any) -> list:
    if isinstance(detection_ism, list):
        return [d for d in detection_ism if isinstance(d, dict)]
    if isinstance(detection_ism, dict):
        dets = detection_ism.get("detections") or detection_ism.get("detection_ism")
        if isinstance(dets, list):
            return [d for d in dets if isinstance(d, dict)]
    return []


def _format_point_info(point: Optional[Tuple[int, int]], *, source: str = "") -> str:
    if point is None:
        return "未选择 point（可选，用于 SAM3 对照预览）"
    x, y = point
    text = f"Point [{x}, {y}]"
    if source:
        text += f"（{source}）"
    return text


def _render_point_preview(
    image: Image.Image,
    point: Optional[Tuple[int, int]],
    *,
    label: str = "point",
) -> Image.Image:
    if point is None:
        return image
    x, y = point
    item = VlmPointItem(
        point_pixel=(x, y),
        label=label,
        point_norm=[float(x), float(y)],
    )
    return render_vlm_points_on_image(image, [item])


def on_ps_image_select(
    evt: gr.SelectData,
    image: Optional[Image.Image],
) -> Tuple[Optional[Tuple[int, int]], str, Optional[Image.Image]]:
    if image is None:
        return None, "请先上传图片", None
    x, y = int(evt.index[0]), int(evt.index[1])
    point = (x, y)
    vis = _render_point_preview(image, point, label="manual")
    return point, _format_point_info(point, source="鼠标点击"), vis


def on_ps_image_change(image: Optional[Image.Image]) -> Tuple[None, str, Optional[Image.Image]]:
    if image is None:
        return None, "请先上传 RGB 图片", None
    return None, "点击图像选择 point（可选，用于 SAM3 对照）", image


def clear_ps_point(image: Optional[Image.Image]) -> Tuple[None, str, Optional[Image.Image]]:
    if image is None:
        return None, "未选择 point", None
    return None, "未选择 point", image


def vlm_get_point_for_sam(
    image: Optional[Image.Image],
    vlm_prompt: str,
    vlm_api_url: str,
    vlm_model: str,
    vlm_temperature: float,
    vlm_timeout_s: float,
    vlm_max_tokens: int,
) -> Tuple[Optional[Tuple[int, int]], str, Optional[Image.Image], str]:
    if image is None:
        return None, "请先上传图片", None, json.dumps(
            {"success": False, "message": "请先上传图片"}, ensure_ascii=False, indent=2
        )

    prompt_text = (vlm_prompt or "").strip()
    if not prompt_text:
        return None, "VLM 提示词不能为空", image, json.dumps(
            {"success": False, "message": "VLM 提示词不能为空"}, ensure_ascii=False, indent=2
        )

    with tempfile.TemporaryDirectory(prefix="vlm_point_sam_") as tmp_dir:
        rgb_path = Path(tmp_dir) / "upload.png"
        image.convert("RGB").save(rgb_path)
        try:
            result = detect_vlm_points(
                rgb_path,
                prompt=prompt_text,
                api_url=(vlm_api_url or "").strip() or DEFAULT_VLM_API_URL,
                model=(vlm_model or "").strip() or DEFAULT_VLM_MODEL,
                timeout_s=float(vlm_timeout_s),
                temperature=float(vlm_temperature),
                max_tokens=max(32, int(vlm_max_tokens)),
            )
        except Exception as exc:
            detail = {
                "success": False,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            return None, str(exc), image, json.dumps(detail, ensure_ascii=False, indent=2)

    if result is None or not result.items:
        return None, "VLM 未返回有效 point", image, json.dumps(
            {
                "success": False,
                "message": "VLM 未返回有效 point",
                "raw_response": result.raw_response if result else None,
            },
            ensure_ascii=False,
            indent=2,
        )

    item = result.items[0]
    point = tuple(item.point_pixel)
    vis = render_vlm_points_on_image(image, [item])
    detail = {
        "success": True,
        "source": "vlm",
        "point_pixel": list(point),
        "point_norm": item.point_norm,
        "label": item.label,
        "raw_response": result.raw_response,
    }
    return point, _format_point_info(point, source="VLM"), vis, json.dumps(
        detail, ensure_ascii=False, indent=2
    )


def _load_camera_and_sensor_depth(
    depth_path: Path,
    camera_path: Path,
) -> tuple[dict, Any, float, Any]:
    with camera_path.open(encoding="utf-8") as f:
        intrinsics_data = json.load(f)
    intrinsics = load_intrinsics_from_dict(intrinsics_data)
    depth_scale = float(intrinsics_data.get("depth_scale", 1.0))
    sensor_depth = load_sensor_depth_from_image(Image.open(depth_path), depth_scale)
    return intrinsics_data, intrinsics, depth_scale, sensor_depth


def _run_fused_depth(
    depth_service: "DepthService",
    image: Image.Image,
    depth_path: Path,
    camera_path: Path,
) -> tuple[Any, Optional[Image.Image], str, Any, dict]:
    intrinsics_data, intrinsics, _, sensor_depth = _load_camera_and_sensor_depth(
        depth_path, camera_path
    )
    t0 = time.perf_counter()
    depth_result = depth_service.predict(
        image,
        intrinsics=intrinsics,
        sensor_depth=sensor_depth,
        source="grasp_ui",
        depth_path=depth_path,
        intrinsics_path=camera_path,
        intrinsics_data=intrinsics_data,
        save_session=False,
    )
    elapsed_s = time.perf_counter() - t0
    if depth_result.fused_depth is None:
        raise RuntimeError("深度融合未产生有效结果，请检查传感器深度与 camera.json")
    fusion_json = json.dumps(
        {
            "success": True,
            "elapsed_s": round(elapsed_s, 3),
            "fusion": depth_result.fusion,
        },
        ensure_ascii=False,
        indent=2,
    )
    return depth_result, depth_result.fused_depth_vis, fusion_json, intrinsics, intrinsics_data, sensor_depth


def _normalize_sam6d_depth_source(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key in ("sensor", "sensor_depth", "raw", "传感器", "传感器深度"):
        return "sensor"
    return "fused"


def _depth_input_vis(depth_result: Any, depth_source: str) -> Optional[Image.Image]:
    source = _normalize_sam6d_depth_source(depth_source)
    if source == "sensor":
        return depth_result.sensor_depth_vis
    return depth_result.fused_depth_vis


def _prepare_sam6d_depth_bundle(
    depth_service: "DepthService",
    image: Image.Image,
    depth_path: Path,
    camera_path: Path,
    depth_source: str,
) -> tuple[Any, str, Any, dict, np.ndarray, Optional[Image.Image], str]:
    """按 depth_source 准备 SAM-6D 深度：sensor 跳过 DA3 融合。"""
    source_key = _normalize_sam6d_depth_source(depth_source)
    intrinsics_data, intrinsics, _, sensor_depth = _load_camera_and_sensor_depth(
        depth_path, camera_path
    )
    valid = np.isfinite(sensor_depth) & (sensor_depth > 0)
    sensor_stats: Dict[str, Any] = {
        "valid_ratio": float(valid.mean()) if valid.any() else 0.0,
    }
    if valid.any():
        sensor_stats["range_mm"] = [
            float(np.nanmin(sensor_depth)),
            float(np.nanmax(sensor_depth)),
        ]

    if source_key == "sensor":
        sensor_vis = depth_to_colormap_with_colorbar(sensor_depth, unit="mm")
        depth_result = SimpleNamespace(
            fused_depth=None,
            sensor_depth_vis=sensor_vis,
            fused_depth_vis=None,
            fusion=None,
        )
        fusion_json = json.dumps(
            {
                "success": True,
                "depth_source": "sensor",
                "fusion_skipped": True,
                "message": "使用原始传感器深度，已跳过 DA3 融合",
                "sensor_stats": sensor_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        logging.getLogger("grasp_pose").info(
            "depth_source=sensor skip fusion valid_ratio=%.3f",
            sensor_stats["valid_ratio"],
        )
        return (
            depth_result,
            fusion_json,
            intrinsics,
            intrinsics_data,
            sensor_depth,
            sensor_vis,
            source_key,
        )

    depth_result, _, fusion_json, intrinsics, intrinsics_data, sensor_depth = _run_fused_depth(
        depth_service, image, depth_path, camera_path
    )
    depth_input_vis = _depth_input_vis(depth_result, source_key)
    return (
        depth_result,
        fusion_json,
        intrinsics,
        intrinsics_data,
        sensor_depth,
        depth_input_vis,
        source_key,
    )


def _sam6d_depth_array(
    depth_source_key: str,
    depth_result: Any,
    sensor_depth: np.ndarray,
) -> np.ndarray:
    if depth_source_key == "sensor":
        return sensor_depth
    if depth_result.fused_depth is None:
        raise RuntimeError("深度融合未产生有效结果，请改用传感器深度或检查输入")
    return depth_result.fused_depth


def _build_grasp_scene_glb(
    depth_service: "DepthService",
    depth_result: Any,
    image: Image.Image,
    intrinsics: Any,
    pem_body: Optional[Dict[str, Any]] = None,
    *,
    scene_depth: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    depth_array = scene_depth if scene_depth is not None else depth_result.fused_depth
    if depth_array is None or intrinsics is None:
        return None, None, None, None
    try:
        poses = extract_pem_poses(pem_body) if pem_body else []
        segment_mask = None
        pem_radius = None
        if pem_body:
            wh = image.size
            segment_mask = extract_best_segmentation_mask(pem_body, wh)
            if segment_mask is not None and poses:
                pem_radius = get_pem_sphere_radius_mm()

        pc_info = build_pointcloud_scene_files(
            depth=depth_array,
            rgb=np.array(image.convert("RGB")),
            intrinsics=intrinsics,
            output_dir=depth_service.output_dir,
            poses=poses or None,
            max_points=depth_service.max_points,
            segment_mask=segment_mask,
            pem_sphere_radius_mm=pem_radius,
        )
        glb_path = str(pc_info["glb_path"])
        ply_path = str(pc_info["ply_path"])
        scene_meta = {
            "point_roles": pc_info.get("point_roles"),
            "segmentation_applied": pc_info.get("segmentation_applied"),
            "pem_sphere_radius_mm": pc_info.get("pem_sphere_radius_mm"),
            "pose_count": pc_info.get("pose_count"),
        }
        logging.getLogger("grasp_pose").info(
            "scene glb built points=%s poses=%s roles=%s path=%s",
            pc_info.get("point_count"),
            pc_info.get("pose_count"),
            pc_info.get("point_roles"),
            glb_path,
        )
        return glb_path, glb_path, ply_path, scene_meta
    except Exception as exc:
        logging.getLogger("grasp_pose").warning("scene glb build failed: %s", exc)
        return None, None, None, None


def run_sam6d_pose_inference(
    depth_service: "DepthService",
    image: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    sam6d_api_url: str,
    sam6d_timeout_s: float,
    genpose2_file_server_url: str,
    sam6d_asset_file_server_url: str,
    depth_source: str,
    sam3_prompt: str,
    sam3_threshold: float,
    sam3_mask_threshold: float,
    det_score_thresh: float,
) -> Sam6dUiOutputs:
    log = logging.getLogger("grasp_pose")
    log.info("run_sam6d_pose_inference start depth_source=%s", depth_source)
    if image is None:
        return _sam6d_error_outputs("请先上传 RGB 图片")

    depth_path = _filepath_from_upload(depth_file)
    camera_path = _filepath_from_upload(camera_file)
    if depth_path is None or not depth_path.is_file():
        return _sam6d_error_outputs("请上传深度图（.png / .exr）")
    if camera_path is None or not camera_path.is_file():
        return _sam6d_error_outputs("请上传 camera.json")

    try:
        (
            depth_result,
            fusion_json,
            intrinsics,
            intrinsics_data,
            sensor_depth,
            depth_input_vis,
            depth_source_key,
        ) = _prepare_sam6d_depth_bundle(
            depth_service, image, depth_path, camera_path, depth_source
        )
        sam6d_depth = _sam6d_depth_array(depth_source_key, depth_result, sensor_depth)
    except Exception as exc:
        log.exception("depth prep failed: %s", exc)
        err = {
            "success": False,
            "stage": "depth_prep",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return _sam6d_error_outputs(json.dumps(err, ensure_ascii=False, indent=2))

    scene_glb, scene_glb_dl, scene_ply, _ = _build_grasp_scene_glb(
        depth_service, depth_result, image, intrinsics, pem_body=None, scene_depth=sam6d_depth
    )

    sam6d_api = (sam6d_api_url or "").strip() or DEFAULT_SAM6D_API_URL
    file_server = (genpose2_file_server_url or "").strip() or DEFAULT_SAM6D_FILE_SERVER_URL
    sam6d_fs = (sam6d_asset_file_server_url or "").strip() or DEFAULT_SAM6D_SAM_FILE_SERVER_URL
    seg = (DEFAULT_SAM6D_SEG_BACKEND or "sam3").strip()
    prompt_text = (sam3_prompt or DEFAULT_SAM3_PROMPT or "").strip()
    asset_kwargs = {
        "file_server_url": file_server,
        "sam6d_file_server_url": sam6d_fs,
        "output_root": DEFAULT_SAM6D_OUTPUT_ROOT,
        "sam6d_output_root": DEFAULT_SAM6D_SAM_OUTPUT_ROOT,
        "api_url": sam6d_api,
    }

    with tempfile.TemporaryDirectory(prefix="sam6d_pose_") as tmp_dir:
        tmp = Path(tmp_dir)
        rgb_path = tmp / "rgb.png"
        sam6d_depth_path = tmp / "sam6d_depth.png"
        camera_out_path = tmp / "camera.json"
        image.convert("RGB").save(rgb_path)
        save_depth_mm_png(sam6d_depth, sam6d_depth_path)
        write_camera_json_for_fused_depth(intrinsics_data, camera_out_path)

        try:
            rgb_wh = image.size
            depth_wh = image_size_wh(sam6d_depth_path)
            if rgb_wh != depth_wh:
                return _sam6d_error_outputs(
                    f"RGB 与深度分辨率不一致: rgb={rgb_wh}, depth={depth_wh}",
                    fused_depth_vis=depth_input_vis,
                    scene_glb=scene_glb,
                    scene_ply=scene_ply,
                )

            t_sam6d = time.perf_counter()
            log.info(
                "calling SAM-6D POST infer depth_source=%s seg_backend=%s sam3_prompt=%r",
                depth_source_key,
                seg,
                prompt_text,
            )
            sam6d_body, sam6d_request = infer_sam6d(
                rgb_path,
                sam6d_depth_path,
                camera_out_path,
                api_url=sam6d_api,
                timeout_s=float(sam6d_timeout_s),
                seg_backend=seg,
                sam3_prompt=prompt_text,
                sam3_threshold=float(sam3_threshold),
                sam3_mask_threshold=float(sam3_mask_threshold),
                det_score_thresh=float(det_score_thresh),
            )
            sam6d_elapsed_s = time.perf_counter() - t_sam6d
            sam6d_request["depth_source"] = depth_source_key
        except Exception as exc:
            log.exception("SAM-6D infer failed: %s", exc)
            err = {
                "success": False,
                "stage": "sam6d",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            return _sam6d_error_outputs(
                json.dumps(err, ensure_ascii=False, indent=2),
                fused_depth_vis=depth_input_vis,
                scene_glb=scene_glb,
                scene_ply=scene_ply,
            )

    sam6d_body = enrich_sam6d_body(sam6d_body, **asset_kwargs)
    scene_glb, scene_glb_dl, scene_ply, scene_point_roles = _build_grasp_scene_glb(
        depth_service, depth_result, image, intrinsics, sam6d_body, scene_depth=sam6d_depth
    )
    pem_poses = extract_pem_poses(sam6d_body)

    ism_combined_vis, ism_vis_url = try_load_sam6d_visualization(
        sam6d_body,
        "vis_ism_path",
        **asset_kwargs,
    )
    pem_vis, pem_vis_url = try_load_sam6d_visualization(
        sam6d_body,
        "vis_pem_path",
        **asset_kwargs,
    )

    ism_dets = _normalize_ism_detections(sam6d_body.get("detection_ism"))
    ism_mask_vis: Optional[Image.Image] = None
    ism_bbox_vis: Optional[Image.Image] = None
    if ism_dets:
        try:
            ism_mask_vis, ism_bbox_vis = render_sam3_mask_bbox_previews(
                image.convert("RGB"),
                ism_dets,
                prompt=prompt_text,
            )
        except Exception as exc:
            log.exception("render ism mask/bbox failed: %s", exc)
    if ism_mask_vis is None:
        ism_mask_vis = ism_combined_vis
    if ism_bbox_vis is None:
        ism_bbox_vis = ism_combined_vis

    depth_colormap, depth_colormap_url, depth_colormap_path = try_load_pem_depth_colormap(
        sam6d_body,
        **asset_kwargs,
    )

    sam6d_health = fetch_sam6d_health(api_url=sam6d_api)

    payload: Dict[str, Any] = {
        "success": True,
        "backend": "sam6d",
        "elapsed_s": round(sam6d_elapsed_s, 3),
        "depth_source": depth_source_key,
        "depth_source_label": "融合深度" if depth_source_key == "fused" else "传感器深度",
        "fusion": depth_result.fusion,
        "fusion_detail": json.loads(fusion_json),
        "scene_glb": scene_glb,
        "scene_pose_count": len(pem_poses),
        "scene_point_roles": scene_point_roles,
        "sam6d_request": sam6d_request,
        "summary": format_pem_summary(sam6d_body),
        "full_response": sam6d_body,
        "cad_path_note": (
            f"CAD 由 SAM-6D 服务端 SAM6D_CAD_PATH 指定（配置参考: {DEFAULT_SAM6D_CAD_PATH}）"
        ),
        "sam6d_health": sam6d_health.get("body"),
        "ism_preview": {
            "num_detections": len(ism_dets),
            "split": "mask / bbox",
            "from_detection_ism": bool(ism_dets),
        },
    }
    if ism_vis_url:
        payload["vis_ism_loaded_from_url"] = ism_vis_url
    if pem_vis_url:
        payload["vis_pem_loaded_from_url"] = pem_vis_url
    if depth_colormap_path:
        payload["depth_colormap_path"] = depth_colormap_path
    if depth_colormap_url:
        payload["depth_colormap_loaded_from_url"] = depth_colormap_url
    if ism_combined_vis is None or pem_vis is None:
        attempted: Dict[str, Any] = {}
        for key in ("vis_ism_path", "vis_pem_path", "detection_pem_path", "detection_ism_path"):
            if sam6d_body.get(key):
                attempted[key] = build_sam6d_asset_urls(
                    str(sam6d_body[key]),
                    file_server_url=file_server,
                    sam6d_file_server_url=sam6d_fs,
                )
        if attempted:
            payload["file_server_urls"] = attempted
        payload["vis_load_note"] = (
            "未能加载 vis_pem/vis_ism。SAM-6D 写入 "
            f"{DEFAULT_SAM6D_SAM_OUTPUT_ROOT}，但 :8003 文件服务根目录是 GenPose2 的 service_outputs，"
            f"新推理结果会 HTTP 404。请启动 :8005 文件服务（见 scripts/start_sam6d_file_server.sh），"
            f"或统一 SAM6D_OUTPUT_ROOT。已尝试 URL 见 file_server_urls。"
        )

    sam6d_json = json.dumps(payload, ensure_ascii=False, indent=2)

    return (
        ism_mask_vis,
        ism_bbox_vis,
        pem_vis,
        depth_input_vis,
        sam6d_json,
        depth_colormap,
        scene_glb,
        scene_glb_dl,
        scene_ply,
    )


def run_sam3_preview(
    image: Optional[Image.Image],
    point_state: Optional[Tuple[int, int]],
    sam_prompt: str,
    sam_api_url: str,
    threshold: float,
    mask_threshold: float,
    sam_timeout_s: float,
) -> Tuple[Optional[Image.Image], str, Optional[Image.Image], str]:
    """可选 SAM3 对照预览（不参与 SAM-6D 位姿 pipeline）。"""
    if image is None:
        err = json.dumps({"success": False, "message": "请先上传 RGB 图片"}, ensure_ascii=False, indent=2)
        return None, err, None, err

    if point_state is None:
        err = json.dumps(
            {"success": False, "message": "SAM3 对照需要 point，请先点击图像或 VLM 获取"},
            ensure_ascii=False,
            indent=2,
        )
        return image, err, None, err

    prompt_text = (sam_prompt or "").strip()
    if not prompt_text:
        err = json.dumps({"success": False, "message": "SAM3 提示词不能为空"}, ensure_ascii=False, indent=2)
        return image, err, None, err

    px, py = int(point_state[0]), int(point_state[1])
    sam_api = (sam_api_url or "").strip() or DEFAULT_SAM3_API_URL

    t0 = time.perf_counter()
    try:
        result, vis = infer_sam3_with_image(
            image,
            prompt=prompt_text,
            api_url=sam_api,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            timeout_s=float(sam_timeout_s),
            return_vis_base64=True,
            points=[(px, py)],
            point_labels=[1],
        )
    except Exception as exc:
        err = json.dumps(
            {
                "success": False,
                "stage": "sam3_preview",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )
        return image, err, None, err

    elapsed_s = time.perf_counter() - t0
    raw = dict(result.raw_response)
    raw.pop("visualization_base64", None)
    summary = {
        "success": True,
        "preview_only": True,
        "elapsed_s": round(elapsed_s, 3),
        "num_detections": result.num_detections,
        "input_point_pixel": [px, py],
        "prompt": result.prompt,
    }
    raw_json = json.dumps(raw, ensure_ascii=False, indent=2)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    raw_vis = decode_sam3_raw_visualization(result, image)
    vis_out = vis if vis is not None else image
    return vis_out, summary_json, raw_vis, raw_json


def build_point_sam_pem_tab(depth_service: "DepthService") -> None:
    """在 ``with gr.Tabs():`` 内调用，添加融合深度 + SAM-6D 位姿 Tab。"""
    with gr.Tab("融合深度 + SAM-6D"):
        gr.Markdown(
            "定位目标料盘并估计 6D 位姿：上传 RGB + 深度 + camera → SAM-6D **`POST /infer`**"
            "（`seg_backend=sam3` 文本分割 + PEM）。"
            "**深度可选融合或原始传感器**（见下方选项）。"
        )
        ps_point_state = gr.State(value=None)
        with gr.Row():
            with gr.Column(scale=1):
                ps_image = gr.Image(
                    type="pil",
                    label="上传 RGB",
                    height=320,
                    interactive=True,
                )
                with gr.Row():
                    ps_depth_file = gr.File(
                        label="传感器深度 depth（uint16 .png，原始深度）",
                        type="filepath",
                        file_types=[".png", ".exr", ".jpg", ".jpeg"],
                    )
                    ps_camera_file = gr.File(
                        label="相机 camera.json",
                        type="filepath",
                        file_types=[".json"],
                    )
                ps_depth_source = gr.Radio(
                    label="SAM-6D 深度输入（传给 /infer 的 depth.png）",
                    choices=[
                        ("融合深度（DA3 + 传感器补洞）", "fused"),
                        ("原始传感器深度（跳过 DA3 融合，更快）", "sensor"),
                    ],
                    value=DEFAULT_SAM6D_DEPTH_SOURCE,
                )
                with gr.Accordion("SAM-6D 高级配置", open=False):
                    ps_sam6d_api = gr.Textbox(
                        label="SAM-6D API URL",
                        value=DEFAULT_SAM6D_API_URL,
                    )
                    ps_sam6d_timeout = gr.Number(
                        label="超时（秒）",
                        value=DEFAULT_SAM6D_TIMEOUT_S,
                        precision=0,
                    )
                    ps_genpose2_file_server = gr.Textbox(
                        label="GenPose2 文件服务 URL（旧结果，可选）",
                        value=DEFAULT_SAM6D_FILE_SERVER_URL,
                        placeholder="http://192.168.100.220:8003",
                    )
                    ps_sam6d_asset_file_server = gr.Textbox(
                        label="SAM-6D 文件服务 URL（vis_ism / vis_pem，:8005）",
                        value=DEFAULT_SAM6D_SAM_FILE_SERVER_URL,
                        placeholder="http://192.168.100.220:8005",
                    )
                    ps_cad_path = gr.Textbox(
                        label="CAD 路径（服务端 SAM6D_CAD_PATH，仅说明）",
                        value=DEFAULT_SAM6D_CAD_PATH,
                        interactive=False,
                    )
                    ps_sam3_prompt = gr.Textbox(
                        label="SAM3 文本提示词 sam3_prompt",
                        value=DEFAULT_SAM3_PROMPT,
                        lines=3,
                    )
                    with gr.Row():
                        ps_sam3_threshold = gr.Slider(
                            label="sam3_threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=DEFAULT_SAM3_THRESHOLD,
                        )
                        ps_sam3_mask_threshold = gr.Slider(
                            label="sam3_mask_threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=DEFAULT_SAM3_MASK_THRESHOLD,
                        )
                    ps_seg_info = gr.Textbox(
                        label="分割后端（固定）",
                        value=f"seg_backend={DEFAULT_SAM6D_SEG_BACKEND or 'sam3'}",
                        interactive=False,
                    )
                    ps_det_score_thresh = gr.Slider(
                        label="PEM 检测分数阈值 det_score_thresh",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=DEFAULT_SAM6D_DET_SCORE_THRESH,
                    )
                ps_infer_btn = gr.Button("运行 SAM-6D 位姿估计", variant="primary")

                with gr.Accordion("SAM3 对照预览（可选，直连 SAM3 服务）", open=False):
                    ps_point_info = gr.Textbox(
                        label="当前 Point",
                        value="点击图像选择 point，或 VLM 获取（SAM3 对照预览用）",
                        interactive=False,
                    )
                    with gr.Row():
                        ps_vlm_btn = gr.Button("VLM 获取 Point")
                        ps_clear_btn = gr.Button("清除 Point")
                    with gr.Accordion("VLM Pointing 提示词", open=False):
                        ps_vlm_prompt = gr.Textbox(
                            label="VLM 提示词",
                            value=DEFAULT_POINT_PROMPT,
                            lines=8,
                        )
                        ps_vlm_api = gr.Textbox(label="VLM API URL", value=DEFAULT_VLM_API_URL)
                        ps_vlm_model = gr.Textbox(label="VLM 模型", value=DEFAULT_VLM_MODEL)
                        ps_vlm_temp = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=DEFAULT_VLM_TEMPERATURE,
                        )
                        ps_vlm_timeout = gr.Number(
                            label="VLM 超时（秒）",
                            value=DEFAULT_VLM_TIMEOUT_S,
                            precision=0,
                        )
                        ps_vlm_max_tokens = gr.Number(label="Max Tokens", value=512, precision=0)
                    with gr.Row():
                        ps_sam_api = gr.Textbox(label="SAM3 API URL（仅对照预览）", value=DEFAULT_SAM3_API_URL)
                        ps_sam_timeout = gr.Number(
                            label="SAM3 超时（秒）",
                            value=DEFAULT_SAM3_TIMEOUT_S,
                            precision=0,
                        )
                    ps_sam3_preview_btn = gr.Button("SAM3 对照预览（直连 SAM3 服务，与位姿参数共用上方 sam3_*）")
                    ps_preview = gr.Image(type="pil", label="Point 预览", height=140)

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                with gr.Row():
                    ps_fused_depth = gr.Image(type="pil", label="SAM-6D 深度输入（伪彩色）", height=180)
                with gr.Row():
                    ps_ism_mask = gr.Image(type="pil", label="SAM-6D ISM mask", height=180)
                    ps_ism_bbox = gr.Image(type="pil", label="SAM-6D ISM bbox", height=180)
                ps_pem_vis = gr.Image(type="pil", label="SAM-6D PEM 位姿可视化", height=220)

        gr.Markdown("### 3D 点云（SAM-6D 深度输入 + 位姿）")
        with gr.Row():
            with gr.Column(scale=4):
                ps_pointcloud_3d = gr.Model3D(
                    label="浏览器预览（GLB）",
                    show_label=True,
                    height=620,
                )
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                ps_glb_download = gr.File(label="GLB", interactive=False)
                ps_ply_download = gr.File(label="PLY", interactive=False)
                gr.Markdown(
                    "**点云着色（推理成功后）**\n"
                    "- 灰色：背景（不参与位姿）\n"
                    "- **橙色**：分割 mask 内、有效深度的点\n"
                    "- **绿色**：mask 内且落在 PEM 球裁剪范围内的点（近似参与位姿配准）\n"
                    "- 坐标轴：红 X / 绿 Y / 蓝 Z，黄点为物体原点\n"
                    "- 深度来源与上方「SAM-6D 深度来源」选项一致"
                )

        gr.Markdown("#### 其他 2D 结果")
        with gr.Row():
            ps_depth_colormap = gr.Image(
                type="pil",
                label="深度伪彩色（若有）",
                height=200,
            )
            ps_sam3_result = gr.Image(type="pil", label="SAM3 对照分割", height=200)
            ps_sam3_raw_vis = gr.Image(type="pil", label="SAM3 原始输出", height=200)

        with gr.Accordion("推理详情（JSON）", open=False):
            ps_sam6d_json = gr.Code(label="SAM-6D 位姿详情", language="json", lines=12)
            ps_sam3_json = gr.Code(label="SAM3 对照详情", language="json", lines=6)
            ps_sam3_raw_json = gr.Code(label="SAM3 原始 JSON", language="json", lines=6)
            ps_vlm_json = gr.Code(label="VLM Point 详情", language="json", lines=4)

        ps_image.select(
            fn=on_ps_image_select,
            inputs=[ps_image],
            outputs=[ps_point_state, ps_point_info, ps_preview],
        )
        ps_image.change(
            fn=on_ps_image_change,
            inputs=[ps_image],
            outputs=[ps_point_state, ps_point_info, ps_preview],
        )
        ps_clear_btn.click(
            fn=clear_ps_point,
            inputs=[ps_image],
            outputs=[ps_point_state, ps_point_info, ps_preview],
        )
        ps_vlm_btn.click(
            fn=vlm_get_point_for_sam,
            inputs=[
                ps_image,
                ps_vlm_prompt,
                ps_vlm_api,
                ps_vlm_model,
                ps_vlm_temp,
                ps_vlm_timeout,
                ps_vlm_max_tokens,
            ],
            outputs=[ps_point_state, ps_point_info, ps_preview, ps_vlm_json],
        )

        def _run_sam6d(*args):
            return run_sam6d_pose_inference(depth_service, *args)

        ps_infer_btn.click(
            fn=_run_sam6d,
            inputs=[
                ps_image,
                ps_depth_file,
                ps_camera_file,
                ps_sam6d_api,
                ps_sam6d_timeout,
                ps_genpose2_file_server,
                ps_sam6d_asset_file_server,
                ps_depth_source,
                ps_sam3_prompt,
                ps_sam3_threshold,
                ps_sam3_mask_threshold,
                ps_det_score_thresh,
            ],
            outputs=[
                ps_ism_mask,
                ps_ism_bbox,
                ps_pem_vis,
                ps_fused_depth,
                ps_sam6d_json,
                ps_depth_colormap,
                ps_pointcloud_3d,
                ps_glb_download,
                ps_ply_download,
            ],
        )

        ps_sam3_preview_btn.click(
            fn=run_sam3_preview,
            inputs=[
                ps_image,
                ps_point_state,
                ps_sam3_prompt,
                ps_sam_api,
                ps_sam3_threshold,
                ps_sam3_mask_threshold,
                ps_sam_timeout,
            ],
            outputs=[ps_sam3_result, ps_sam3_json, ps_sam3_raw_vis, ps_sam3_raw_json],
        )

        gr.Markdown(
            f"""
            **流程**
            1. 上传 **RGB**、**传感器深度**、**camera.json**
            2. 选择 **深度输入**：融合 / **原始传感器**（传感器模式跳过 DA3，更快）
            3. 点击「运行 SAM-6D 位姿估计」

            **说明**
            - **原始传感器深度**：直接上传 depth.png（mm），保留空洞，适合对比融合效果
            - **融合深度**：DA3 估计 + 传感器对齐补洞
            - **SAM-6D**：`POST /infer`，multipart 上传 `rgb` + **depth** + `camera`，form 字段 `seg_backend=sam3`、`sam3_prompt`、`sam3_threshold`、`sam3_mask_threshold`、`det_score_thresh`
            - **3D 预览**：与所选深度来源一致的点云 + SAM-6D 6D 位姿坐标轴
            - **SAM3 对照**：可选，直连 SAM3 HTTP 服务预览分割，参数与位姿推理共用上方 sam3_* 字段
            - 可视化 PNG 经 HTTP 拉取：**SAM-6D 新结果**用 `:8005`（`scripts/start_sam6d_file_server.sh`），旧 GenPose2 结果可用 `:8003`
            - 默认 SAM-6D：`{DEFAULT_SAM6D_API_URL}`；默认 CAD：`{DEFAULT_SAM6D_CAD_PATH}`
            """
        )


build_sam6d_pose_tab = build_point_sam_pem_tab
