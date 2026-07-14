"""
SAM-6D 位姿估计 HTTP 客户端（multipart POST /infer：rgb + depth + camera，无 mask）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from src.grasp.settings import (
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM6D_API_URL,
    DEFAULT_SAM6D_DET_SCORE_THRESH,
    DEFAULT_SAM6D_FILE_SERVER_URL,
    DEFAULT_SAM6D_OUTPUT_ROOT,
    DEFAULT_SAM6D_SAM_FILE_SERVER_URL,
    DEFAULT_SAM6D_SAM_OUTPUT_ROOT,
    DEFAULT_SAM6D_SEG_BACKEND,
    DEFAULT_SAM6D_SEGMENTOR_MODEL,
    DEFAULT_SAM6D_TIMEOUT_S,
)


def pem_health_url(api_url: str) -> str:
    parsed = urlparse(api_url)
    path = parsed.path or "/infer"
    if path.endswith("/infer"):
        path = path[: -len("/infer")] + "/health"
    else:
        path = "/health"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def check_sam6d_health(
    *,
    api_url: Optional[str] = None,
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    url = api_url or DEFAULT_SAM6D_API_URL
    health_url = pem_health_url(url)
    try:
        resp = requests.get(health_url, timeout=timeout_s)
        body = resp.json() if resp.content else {}
        return {
            "ok": resp.status_code == 200 and body.get("status") == "ok",
            "api_url": url,
            "health_url": health_url,
            "status_code": resp.status_code,
            "body": body,
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "api_url": url,
            "health_url": health_url,
            "error": str(exc),
        }


check_pem_health = check_sam6d_health


def _mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image/png"
    if suffix == ".exr":
        return "application/octet-stream"
    if suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def _log(msg: str) -> None:
    logging.getLogger("sam6d_infer").info(msg)


_health_cache: Optional[Dict[str, Any]] = None


def fetch_sam6d_health(
    *,
    api_url: Optional[str] = None,
    timeout_s: float = 5.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    global _health_cache
    if use_cache and _health_cache is not None:
        return _health_cache

    url = api_url or DEFAULT_SAM6D_API_URL
    health_url = pem_health_url(url)
    try:
        resp = requests.get(health_url, timeout=timeout_s)
        body = resp.json() if resp.content else {}
        result = {
            "ok": resp.status_code == 200 and body.get("status") == "ok",
            "api_url": url,
            "health_url": health_url,
            "status_code": resp.status_code,
            "body": body,
        }
    except requests.RequestException as exc:
        result = {
            "ok": False,
            "api_url": url,
            "health_url": health_url,
            "error": str(exc),
        }

    if use_cache:
        _health_cache = result
    return result


def resolve_sam6d_relative_path(path_str: Optional[str]) -> Optional[str]:
    """从绝对路径提取 request_id/... 相对段。"""
    if not path_str:
        return None
    path_str = str(path_str).strip()
    if "service_outputs/" in path_str:
        return path_str.split("service_outputs/", 1)[1].lstrip("/")
    parts = Path(path_str).parts
    for idx, part in enumerate(parts):
        if len(part) >= 17 and part[:8].isdigit() and part[8] == "_":
            return "/".join(parts[idx:])
    return None


def build_sam6d_asset_roots(
    *,
    output_root: Optional[str] = None,
    sam6d_output_root: Optional[str] = None,
    api_url: Optional[str] = None,
) -> List[str]:
    roots: List[str] = []
    for candidate in (
        output_root or DEFAULT_SAM6D_OUTPUT_ROOT,
        sam6d_output_root or DEFAULT_SAM6D_SAM_OUTPUT_ROOT,
    ):
        text = (candidate or "").strip().rstrip("/")
        if text and text not in roots:
            roots.append(text)

    health = fetch_sam6d_health(api_url=api_url)
    health_root = (health.get("body") or {}).get("output_root")
    if isinstance(health_root, str):
        text = health_root.strip().rstrip("/")
        if text and text not in roots:
            roots.append(text)
    return roots


def build_sam6d_file_servers(
    *,
    file_server_url: Optional[str] = None,
    sam6d_file_server_url: Optional[str] = None,
) -> List[str]:
    """SAM-6D 专用文件服务优先（新结果在 mui output_root）。"""
    servers: List[str] = []
    for candidate in (
        sam6d_file_server_url or DEFAULT_SAM6D_SAM_FILE_SERVER_URL,
        file_server_url or DEFAULT_SAM6D_FILE_SERVER_URL,
    ):
        text = (candidate or "").strip().rstrip("/")
        if text and text not in servers:
            servers.append(text)
    return servers


def _local_path_candidates(path_str: Optional[str], output_roots: List[str]) -> List[Path]:
    if not path_str:
        return []
    rel = resolve_sam6d_relative_path(path_str)
    candidates: List[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    _add(Path(str(path_str)).expanduser())
    if rel:
        for root in output_roots:
            _add(Path(root) / rel)
    return candidates


def _file_upload_meta(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "mime": _mime_for_path(path),
    }


def _parse_error_body(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:500]
    if isinstance(body, dict):
        detail = body.get("detail")
        if detail is not None:
            return str(detail)
        err = body.get("error")
        if err is not None:
            return str(err)
    return resp.text[:500]


def write_camera_json_for_fused_depth(
    intrinsics_data: Dict[str, Any],
    out_path: Path,
) -> Path:
    """写入 camera.json，depth_scale=1.0（融合深度 PNG 像素值即为 mm）。"""
    payload = dict(intrinsics_data)
    payload["depth_scale"] = 1.0
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def infer_sam6d(
    rgb_path: Path,
    depth_path: Path,
    camera_path: Path,
    *,
    api_url: Optional[str] = None,
    timeout_s: float = DEFAULT_SAM6D_TIMEOUT_S,
    seg_backend: Optional[str] = None,
    det_score_thresh: Optional[float] = None,
    sam3_prompt: Optional[str] = None,
    sam3_threshold: Optional[float] = None,
    sam3_mask_threshold: Optional[float] = None,
    segmentor_model: Optional[str] = None,
    yolo_weights: Optional[str] = None,
    yolo_conf: Optional[float] = None,
    yolo_imgsz: Optional[int] = None,
    yolo_class_id: Optional[int] = None,
    mask_score: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    上传 rgb + depth + camera，调用 SAM-6D ``POST /infer``（multipart form-data）。

    默认 ``seg_backend=sam3``，与 curl ``-F seg_backend=sam3 -F sam3_prompt=...`` 一致。
    控制参数走 form 字段，不走 query string。

    :return: (响应 body, 请求元数据)
    """
    rgb_path = rgb_path.expanduser().resolve()
    depth_path = depth_path.expanduser().resolve()
    camera_path = camera_path.expanduser().resolve()

    for label, p in (("rgb", rgb_path), ("depth", depth_path), ("camera", camera_path)):
        if not p.is_file():
            raise FileNotFoundError(f"SAM-6D {label} file not found: {p}")

    url = (api_url or DEFAULT_SAM6D_API_URL).strip()
    effective_backend = (seg_backend or DEFAULT_SAM6D_SEG_BACKEND or "sam3").strip()
    form_data: Dict[str, str] = {"seg_backend": effective_backend}

    if det_score_thresh is not None:
        form_data["det_score_thresh"] = str(float(det_score_thresh))

    if effective_backend == "sam3":
        prompt = (sam3_prompt or DEFAULT_SAM3_PROMPT or "").strip()
        if not prompt:
            raise ValueError("seg_backend=sam3 时 sam3_prompt 不能为空")
        form_data["sam3_prompt"] = prompt
        form_data["sam3_threshold"] = str(
            float(sam3_threshold if sam3_threshold is not None else DEFAULT_SAM3_THRESHOLD)
        )
        form_data["sam3_mask_threshold"] = str(
            float(
                sam3_mask_threshold
                if sam3_mask_threshold is not None
                else DEFAULT_SAM3_MASK_THRESHOLD
            )
        )
    elif effective_backend == "sam6d_ism":
        form_data["segmentor_model"] = (
            segmentor_model or DEFAULT_SAM6D_SEGMENTOR_MODEL or "sam"
        ).strip()
    elif effective_backend == "yolo_seg":
        if yolo_weights:
            form_data["yolo_weights"] = yolo_weights
        if yolo_conf is not None:
            form_data["yolo_conf"] = str(float(yolo_conf))
        if yolo_imgsz is not None:
            form_data["yolo_imgsz"] = str(int(yolo_imgsz))
        if yolo_class_id is not None:
            form_data["yolo_class_id"] = str(int(yolo_class_id))
    elif effective_backend == "user_mask":
        if mask_score is not None:
            form_data["mask_score"] = str(float(mask_score))

    request_meta: Dict[str, Any] = {
        "url": url,
        "method": "POST",
        "form_fields": {
            **form_data,
            "rgb": _file_upload_meta(rgb_path),
            "depth": _file_upload_meta(depth_path),
            "camera": _file_upload_meta(camera_path),
        },
        "mask_uploaded": False,
        "backend": "sam6d",
        "seg_backend": effective_backend,
    }

    _log(
        f"[sam6d_infer] POST {url} "
        f"multipart form={form_data} files=[rgb,depth,camera]"
    )

    with (
        open(rgb_path, "rb") as rgb_f,
        open(depth_path, "rb") as depth_f,
        open(camera_path, "rb") as camera_f,
    ):
        files = {
            "rgb": (rgb_path.name, rgb_f, _mime_for_path(rgb_path)),
            "depth": (depth_path.name, depth_f, _mime_for_path(depth_path)),
            "camera": (camera_path.name, camera_f, _mime_for_path(camera_path)),
        }
        try:
            resp = requests.post(url, files=files, data=form_data, timeout=float(timeout_s))
        except requests.RequestException as exc:
            raise RuntimeError(f"SAM-6D 服务不可达: {url}") from exc

    request_meta["http_status"] = resp.status_code

    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(
            f"SAM-6D invalid JSON: status={resp.status_code} body={resp.text[:500]}"
        ) from None

    if resp.status_code != 200:
        err = _parse_error_body(resp)
        _log(f"[sam6d_infer] failed status={resp.status_code} error={err}")
        raise RuntimeError(f"SAM-6D infer failed: status={resp.status_code} error={err}")

    if not isinstance(body, dict):
        raise RuntimeError(f"SAM-6D response must be object, got {type(body).__name__}")

    body = enrich_sam6d_body(body)

    _log(
        f"[sam6d_infer] ok score={body.get('score')} "
        f"xyz_mm={body.get('xyz_mm')} "
        f"result_dir={body.get('result_dir')}"
    )
    request_meta["response_summary"] = {
        "score": body.get("score"),
        "xyz_mm": body.get("xyz_mm"),
        "result_dir": body.get("result_dir"),
        "timing": body.get("timing"),
    }
    return body, request_meta


def _local_path_readable(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _load_json_file(path_str: Optional[str]) -> Optional[Any]:
    if not path_str:
        return None
    path = Path(str(path_str)).expanduser()
    if not _local_path_readable(path):
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log(f"[sam6d_infer] cannot load json {path}: {exc}")
        return None


def try_load_json_via_path(
    path_str: Optional[str],
    *,
    file_server_url: Optional[str] = None,
    sam6d_file_server_url: Optional[str] = None,
    output_root: Optional[str] = None,
    sam6d_output_root: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout_s: float = 15.0,
    log_label: str = "json",
) -> Optional[Any]:
    """从本地路径或多个 HTTP 文件服务加载 JSON。"""
    output_roots = build_sam6d_asset_roots(
        output_root=output_root,
        sam6d_output_root=sam6d_output_root,
        api_url=api_url,
    )
    for local_path in _local_path_candidates(path_str, output_roots):
        data = _load_json_file(str(local_path))
        if data is not None:
            return data

    rel = resolve_sam6d_relative_path(path_str)
    if not rel:
        return None

    for server in build_sam6d_file_servers(
        file_server_url=file_server_url,
        sam6d_file_server_url=sam6d_file_server_url,
    ):
        url = f"{server}/{rel.lstrip('/')}"
        try:
            resp = requests.get(url, timeout=float(timeout_s))
            resp.raise_for_status()
            _log(f"[sam6d_infer] fetch {log_label} -> {url}")
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            _log(f"[sam6d_infer] cannot load json url {url}: {exc}")
    return None


def enrich_sam6d_body(
    body: Dict[str, Any],
    *,
    file_server_url: Optional[str] = None,
    sam6d_file_server_url: Optional[str] = None,
    output_root: Optional[str] = None,
    sam6d_output_root: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """若响应含 detection_pem_path，尝试加载 detections_pem 供位姿解析。"""
    if body.get("detections_pem"):
        return body

    load_kwargs = {
        "file_server_url": file_server_url,
        "sam6d_file_server_url": sam6d_file_server_url,
        "output_root": output_root,
        "sam6d_output_root": sam6d_output_root,
        "api_url": api_url,
    }

    pem_data = try_load_json_via_path(
        body.get("detection_pem_path"),
        log_label="detection_pem",
        **load_kwargs,
    )
    if isinstance(pem_data, list):
        body["detections_pem"] = pem_data
    elif isinstance(pem_data, dict):
        dets = pem_data.get("detections") or pem_data.get("detections_pem")
        if isinstance(dets, list):
            body["detections_pem"] = dets

    ism_data = try_load_json_via_path(
        body.get("detection_ism_path"),
        log_label="detection_ism",
        **load_kwargs,
    )
    if ism_data is not None:
        body["detection_ism"] = ism_data
    return body


def image_size_wh(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def try_load_local_image(path_str: Optional[str]) -> Optional[Image.Image]:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not _local_path_readable(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        _log(f"[sam6d_infer] cannot load local image {path}: {exc}")
        return None


def pem_path_to_file_server_url(
    path_str: Optional[str],
    *,
    file_server_url: str,
    output_root: str = DEFAULT_SAM6D_OUTPUT_ROOT,
) -> Optional[str]:
    """将 SAM-6D 服务端绝对路径转为 http.server 可访问 URL。"""
    if not path_str or not (file_server_url or "").strip():
        return None

    path_str = path_str.strip()
    root = output_root.rstrip("/")
    rel: Optional[str] = None

    if path_str.startswith(root + "/"):
        rel = path_str[len(root) + 1 :]
    elif "service_outputs/" in path_str:
        rel = path_str.split("service_outputs/", 1)[1]
    else:
        # 兜底：从路径中提取 request_id/... 相对段
        parts = Path(path_str).parts
        for idx, part in enumerate(parts):
            if len(part) >= 17 and part[0:8].isdigit() and part[8] == "_":
                rel = "/".join(parts[idx:])
                break

    if not rel:
        return None

    return f"{file_server_url.rstrip('/')}/{rel.lstrip('/')}"


def try_load_image_from_url(url: str, *, timeout_s: float = 15.0) -> Optional[Image.Image]:
    import io

    try:
        resp = requests.get(url, timeout=float(timeout_s))
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        _log(f"[sam6d_infer] cannot load url {url}: {exc}")
        return None


def try_load_pem_image_path(
    path_str: Optional[str],
    *,
    file_server_url: Optional[str] = None,
    sam6d_file_server_url: Optional[str] = None,
    output_root: Optional[str] = None,
    sam6d_output_root: Optional[str] = None,
    api_url: Optional[str] = None,
    log_label: str = "image",
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """按绝对路径加载 SAM-6D 输出图（多本地根目录 + 多 HTTP 文件服务）。"""
    if not path_str:
        return None, None

    output_roots = build_sam6d_asset_roots(
        output_root=output_root,
        sam6d_output_root=sam6d_output_root,
        api_url=api_url,
    )
    for local_path in _local_path_candidates(path_str, output_roots):
        img = try_load_local_image(str(local_path))
        if img is not None:
            return img, None

    rel = resolve_sam6d_relative_path(path_str)
    if not rel:
        return None, None

    for server in build_sam6d_file_servers(
        file_server_url=file_server_url,
        sam6d_file_server_url=sam6d_file_server_url,
    ):
        url = f"{server}/{rel.lstrip('/')}"
        _log(f"[sam6d_infer] fetch {log_label} -> {url}")
        img = try_load_image_from_url(url)
        if img is not None:
            return img, url
    return None, None


def try_load_sam6d_visualization(
    pem_body: Dict[str, Any],
    key: str,
    *,
    file_server_url: Optional[str] = None,
    sam6d_file_server_url: Optional[str] = None,
    output_root: Optional[str] = None,
    sam6d_output_root: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """加载指定可视化图（vis_ism_path / vis_pem_path）。"""
    path_str = pem_body.get(key)
    if not path_str:
        return None, None
    return try_load_pem_image_path(
        str(path_str),
        file_server_url=file_server_url,
        sam6d_file_server_url=sam6d_file_server_url,
        output_root=output_root,
        sam6d_output_root=sam6d_output_root,
        api_url=api_url,
        log_label=key,
    )


def try_load_pem_visualization(
    pem_body: Dict[str, Any],
    *,
    file_server_url: Optional[str] = None,
    output_root: str = DEFAULT_SAM6D_OUTPUT_ROOT,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """加载 PEM 位姿可视化图（优先 vis_pem，其次 vis_ism）。"""
    file_server = (file_server_url or "").strip()
    for key in ("vis_pem_path", "vis_ism_path"):
        img, url = try_load_sam6d_visualization(
            pem_body,
            key,
            file_server_url=file_server,
            output_root=output_root,
        )
        if img is not None:
            return img, url
    return None, None


def resolve_depth_colormap_candidates(pem_body: Dict[str, Any]) -> List[str]:
    """解析深度伪彩色候选路径（兼容 depth_colormap.png / depth_colormap_path.png）。"""
    candidates: List[str] = []
    seen: set[str] = set()

    def _add(path: Optional[str]) -> None:
        if not path:
            return
        p = str(path).strip()
        if p and p not in seen:
            seen.add(p)
            candidates.append(p)

    for key in ("depth_colormap_path", "depth_colormap"):
        _add(pem_body.get(key))

    for anchor_key in ("vis_pem_path", "vis_ism_path", "detection_pem_path"):
        anchor = pem_body.get(anchor_key)
        if not anchor:
            continue
        parent = Path(str(anchor)).parent
        _add(str(parent / "depth_colormap_path.png"))
        _add(str(parent / "depth_colormap.png"))
        break

    result_dir = pem_body.get("result_dir")
    if result_dir:
        sam6d = Path(str(result_dir)) / "sam6d_results"
        _add(str(sam6d / "depth_colormap_path.png"))
        _add(str(sam6d / "depth_colormap.png"))

    return candidates


def resolve_depth_colormap_path(pem_body: Dict[str, Any]) -> Optional[str]:
    candidates = resolve_depth_colormap_candidates(pem_body)
    return candidates[0] if candidates else None


def try_load_pem_depth_colormap(
    pem_body: Dict[str, Any],
    *,
    file_server_url: Optional[str] = None,
    sam6d_file_server_url: Optional[str] = None,
    output_root: Optional[str] = None,
    sam6d_output_root: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Tuple[Optional[Image.Image], Optional[str], Optional[str]]:
    load_kwargs = {
        "file_server_url": file_server_url,
        "sam6d_file_server_url": sam6d_file_server_url,
        "output_root": output_root,
        "sam6d_output_root": sam6d_output_root,
        "api_url": api_url,
    }
    for path_str in resolve_depth_colormap_candidates(pem_body):
        img, url = try_load_pem_image_path(
            path_str,
            log_label="depth_colormap",
            **load_kwargs,
        )
        if img is not None:
            return img, url, path_str
    return None, None, resolve_depth_colormap_path(pem_body)


def build_sam6d_asset_urls(
    path_str: Optional[str],
    *,
    file_server_url: Optional[str] = None,
    sam6d_file_server_url: Optional[str] = None,
) -> List[str]:
    rel = resolve_sam6d_relative_path(path_str)
    if not rel:
        return []
    return [
        f"{server}/{rel.lstrip('/')}"
        for server in build_sam6d_file_servers(
            file_server_url=file_server_url,
            sam6d_file_server_url=sam6d_file_server_url,
        )
    ]


def extract_best_segmentation_mask(
    body: Dict[str, Any],
    image_size_wh: tuple[int, int],
) -> Optional[np.ndarray]:
    """从 SAM-6D ``detection_ism`` 取 score 最高的分割 mask（H×W bool）。"""
    from src.grasp.sam3 import _decode_detection_mask

    detections = body.get("detection_ism")
    if not isinstance(detections, list) or not detections:
        return None

    valid_dets = [d for d in detections if isinstance(d, dict) and "segmentation" in d]
    if not valid_dets:
        return None

    best = max(valid_dets, key=lambda d: float(d.get("score", 0.0)))
    try:
        return _decode_detection_mask(best, image_size_wh)
    except Exception as exc:
        _log(f"[sam6d_infer] decode ism mask failed: {exc}")
        return None


def get_pem_sphere_radius_mm() -> float:
    """CAD 包围球半径（mm），用于可视化 PEM 深度裁剪范围。"""
    try:
        from src.grasp.cad import cad_bounding_radius_mm

        return cad_bounding_radius_mm()
    except Exception as exc:
        _log(f"[sam6d_infer] cad radius fallback: {exc}")
        return 115.0


def extract_pem_poses(body: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """从 SAM-6D 响应解析相机系位姿（平移 mm + 旋转矩阵）。"""
    from src.depth.pointcloud import euler_zyx_to_rotation_matrix

    poses: List[Tuple[np.ndarray, np.ndarray]] = []

    def _append(t_val: Any, r_val: Any) -> None:
        if t_val is None or r_val is None:
            return
        poses.append(
            (
                np.asarray(t_val, dtype=np.float64),
                np.asarray(r_val, dtype=np.float64),
            )
        )

    for det in body.get("detections_pem") or []:
        if not isinstance(det, dict):
            continue
        _append(det.get("t") or det.get("xyz_mm"), det.get("R"))

    if not poses and body.get("xyz_mm"):
        rotation = body.get("rotation_matrix")
        if rotation is None and body.get("rotation_euler_zyx_rad"):
            rotation = euler_zyx_to_rotation_matrix(body["rotation_euler_zyx_rad"])
        _append(body.get("xyz_mm"), rotation)

    return poses


def format_pem_summary(body: Dict[str, Any]) -> Dict[str, Any]:
    """提取位姿摘要，便于界面 JSON 展示。"""
    summary: Dict[str, Any] = {
        "success": True,
        "backend": "sam6d",
        "score": body.get("score"),
        "xyz_mm": body.get("xyz_mm"),
        "rotation_euler_zyx_rad": body.get("rotation_euler_zyx_rad"),
        "xyzrxryrz": body.get("xyzrxryrz"),
        "xyzrxryrz_unit": body.get("xyzrxryrz_unit"),
        "result_dir": body.get("result_dir"),
        "vis_pem_path": body.get("vis_pem_path"),
        "vis_ism_path": body.get("vis_ism_path"),
        "detection_ism_path": body.get("detection_ism_path"),
        "detection_pem_path": body.get("detection_pem_path"),
        "depth_colormap_path": body.get("depth_colormap_path"),
        "timing": body.get("timing"),
    }
    if body.get("detections_pem"):
        summary["detections_pem"] = body["detections_pem"]
    if body.get("detection_ism") is not None:
        summary["detection_ism"] = body["detection_ism"]
    return summary
