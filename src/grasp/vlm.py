"""
VLM ROI + SAM3 原图分割 + ROI 交集筛选。

流程见 ``vlm_seg.md``：
  ① 原图 ``rgb`` 跑 ``SAM3`` 得到实例
  ② 原图 ``rgb`` 跑 ``VLM`` 得到 ROI
  ③ 仅保留与 ROI 交集最大的 instance，送入 FoundationPose
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from src.grasp.settings import (
    DEFAULT_VLM_API_URL,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TEMPERATURE,
    DEFAULT_VLM_TIMEOUT_S,
)
from src.grasp.sam3 import (
    DEFAULT_SAM3_PROMPT,
    Sam3SegmentationResult,
    get_instance_bool_masks,
    run_sam3_segmentation,
    visualize_sam3_ism,
    visualize_sam3_mask_exr,
)
from src.grasp.paths import PROMPT_DIR

DEFAULT_VLM_ROI_MIN_AREA_PX = 100
DEFAULT_VLM_ROI_MARGIN_PX = 10
DEFAULT_VLM_MIN_INTERSECTION_PX = 1
_PROMPT_DIR = PROMPT_DIR
_DEFAULT_ROI_PROMPT_FILE = _PROMPT_DIR / "plastic tray.txt"


def _load_default_vlm_prompt() -> str:
    if _DEFAULT_ROI_PROMPT_FILE.is_file():
        return _DEFAULT_ROI_PROMPT_FILE.read_text(encoding="utf-8").strip()
    return (
        "Detect the single white plastic tray above the green square slot marker. "
        'Return [{"bbox_2d": [x1,y1,x2,y2], "label": "white_tray_above_green_marker"}].'
    )


DEFAULT_VLM_PROMPT = _load_default_vlm_prompt()


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _copy_text_file(src: Path, dst: Path) -> Path:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _copy_binary_file(src: Path, dst: Path) -> Path:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        dst.write_bytes(src.read_bytes())
    return dst


def _copy_image_file(src: Path, dst: Path) -> Path:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_file():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src == dst:
        return dst
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {src}")
    ok = cv2.imwrite(str(dst), image)
    if not ok:
        raise RuntimeError(f"failed to write image: {dst}")
    return dst


def _save_json(path: Path, payload: Any) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _vlm_api_url() -> str:
    return _env_first("GENPOSE2_VLM_API_URL", default=DEFAULT_VLM_API_URL)


def _vlm_model() -> str:
    return _env_first("GENPOSE2_VLM_MODEL", default=DEFAULT_VLM_MODEL)


def _vlm_prompt() -> str:
    return _env_first("GENPOSE2_VLM_PROMPT", default=DEFAULT_VLM_PROMPT)


def _vlm_roi_margin_px() -> int:
    return max(
        0,
        int(
            _env_first(
                "GENPOSE2_VLM_ROI_MARGIN_PX",
                default=str(DEFAULT_VLM_ROI_MARGIN_PX),
            )
        ),
    )


def _vlm_roi_min_area() -> int:
    return max(
        1,
        int(
            _env_first(
                "GENPOSE2_VLM_ROI_MIN_AREA_PX",
                default=str(DEFAULT_VLM_ROI_MIN_AREA_PX),
            )
        ),
    )


def _vlm_min_intersection_px() -> int:
    return max(
        1,
        int(
            _env_first(
                "GENPOSE2_VLM_MIN_INTERSECTION_PX",
                default=str(DEFAULT_VLM_MIN_INTERSECTION_PX),
            )
        ),
    )


def use_vlm_roi_filter() -> bool:
    return _env_first(
        "GENPOSE2_USE_VLM_ROI_FILTER",
        "GENPOSE2_USE_VLM_ROI",
        default="1",
    ).lower() not in ("0", "false", "no")


def use_vlm_roi() -> bool:
    """兼容旧接口名。"""
    return use_vlm_roi_filter()


def seg_backend() -> str:
    """兼容旧接口，新的方案固定为 ``sam3``。"""
    return "sam3"


def clip_int(v: float, low: int, high: int) -> int:
    return int(np.clip(round(v), low, high))


def bbox_to_pixel(
    bbox: List[float],
    W: int,
    H: int,
    mode: str = "norm1000",
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if mode == "pixel":
        px1 = clip_int(x1, 0, W - 1)
        py1 = clip_int(y1, 0, H - 1)
        px2 = clip_int(x2, 0, W - 1)
        py2 = clip_int(y2, 0, H - 1)
    elif mode == "norm1000":
        px1 = clip_int(x1 / 1000.0 * (W - 1), 0, W - 1)
        py1 = clip_int(y1 / 1000.0 * (H - 1), 0, H - 1)
        px2 = clip_int(x2 / 1000.0 * (W - 1), 0, W - 1)
        py2 = clip_int(y2 / 1000.0 * (H - 1), 0, H - 1)
    else:
        if max(x1, y1, x2, y2) <= 1000.0 and (W > 1100 or H > 1100):
            px1 = clip_int(x1 / 1000.0 * (W - 1), 0, W - 1)
            py1 = clip_int(y1 / 1000.0 * (H - 1), 0, H - 1)
            px2 = clip_int(x2 / 1000.0 * (W - 1), 0, W - 1)
            py2 = clip_int(y2 / 1000.0 * (H - 1), 0, H - 1)
        else:
            px1 = clip_int(x1, 0, W - 1)
            py1 = clip_int(y1, 0, H - 1)
            px2 = clip_int(x2, 0, W - 1)
            py2 = clip_int(y2, 0, H - 1)
    xx1, xx2 = sorted([px1, px2])
    yy1, yy2 = sorted([py1, py2])
    return xx1, yy1, xx2, yy2


def point_to_pixel(
    point: List[float],
    W: int,
    H: int,
    mode: str = "norm1000",
) -> Tuple[int, int]:
    x, y = point
    if mode == "pixel":
        px = clip_int(x, 0, W - 1)
        py = clip_int(y, 0, H - 1)
    elif mode == "norm1000":
        px = clip_int(x / 1000.0 * (W - 1), 0, W - 1)
        py = clip_int(y / 1000.0 * (H - 1), 0, H - 1)
    else:
        if max(x, y) <= 1000.0 and (W > 1100 or H > 1100):
            px = clip_int(x / 1000.0 * (W - 1), 0, W - 1)
            py = clip_int(y / 1000.0 * (H - 1), 0, H - 1)
        else:
            px = clip_int(x, 0, W - 1)
            py = clip_int(y, 0, H - 1)
    return px, py


def _image_to_base64(file_path: Path) -> str:
    import base64

    with file_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _parse_vlm_detection_list(generated_text: str) -> List[Dict[str, Any]]:
    text = _strip_json_fence(generated_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    dets: List[Dict[str, Any]] = []
    for det in data:
        if not isinstance(det, dict) or "bbox_2d" not in det:
            continue
        bbox = det["bbox_2d"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        dets.append(det)
    return dets


def _parse_vlm_detections(generated_text: str) -> Optional[Dict[str, Any]]:
    dets = _parse_vlm_detection_list(generated_text)
    return dets[0] if dets else None


def _parse_vlm_point_list(generated_text: str) -> List[Dict[str, Any]]:
    text = _strip_json_fence(generated_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        if "point_2d" in data or "point" in data:
            data = [data]
        else:
            return []

    if not isinstance(data, list):
        return []

    points: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        point = item.get("point_2d") or item.get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        points.append(item)
    return points


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    margin_px: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = clip_int(x1 - margin_px, 0, width - 1)
    y1 = clip_int(y1 - margin_px, 0, height - 1)
    x2 = clip_int(x2 + margin_px, 0, width - 1)
    y2 = clip_int(y2 + margin_px, 0, height - 1)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _read_image_size(rgb_path: Path) -> Tuple[int, int]:
    with Image.open(rgb_path) as img:
        return img.size  # (W, H)


def _load_detection_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"empty or invalid detection_ism.json: {path}")
    return [det for det in data if isinstance(det, dict)]


def _save_instance_id_mask_png(mask_path: Path, composite: np.ndarray) -> Path:
    mask_path = mask_path.expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.suffix.lower() == ".exr":
        mask_path = mask_path.with_suffix(".png")
    ok = cv2.imwrite(str(mask_path), composite)
    if not ok:
        raise RuntimeError(f"failed to write instance mask: {mask_path}")
    return mask_path


@dataclass
class VlmRoiResult:
    bbox_pixel: Tuple[int, int, int, int]
    label: str
    raw_response: str
    bbox_norm: Optional[List[float]] = None


@dataclass
class VlmDetectionItem:
    bbox_pixel: Tuple[int, int, int, int]
    label: str
    bbox_norm: List[float]


@dataclass
class VlmDetectionsResult:
    items: List[VlmDetectionItem]
    raw_response: str


@dataclass
class VlmPointItem:
    point_pixel: Tuple[int, int]
    label: str
    point_norm: List[float]


@dataclass
class VlmPointsResult:
    items: List[VlmPointItem]
    raw_response: str


DEFAULT_VLM_LABEL_COLORS: Dict[str, str] = {
    "slot_marker": "#00cc44",
    "electronic_reel": "#ff3333",
    "blue_dot_below_empty_slot": "#3366ff",
    "empty_tray_slot_above_blue_dot": "#ff8800",
    "white_tray_above_blue_dot": "#ff3333",
    "white_tray_above_green_marker": "#ff3333",
    "green_square_marker": "#00cc44",
}


@dataclass
class FilteredInstancesResult:
    sam3: Sam3SegmentationResult
    raw_detection_json: Path
    filtered_detection_json: Path
    filtered_detection_json_alias: Path
    mask_path: Path
    vis_ism_path: Optional[Path]
    vis_mask_path: Optional[Path]
    kept_instance_ids: List[int]
    kept_source_indices: List[int]
    intersection_pixels: Dict[int, int]
    raw_intersection_pixels: Dict[int, int]
    roi_bbox_used: Tuple[int, int, int, int]


@dataclass
class VlmSam3FilterResult:
    sam3: Sam3SegmentationResult
    raw_sam3: Sam3SegmentationResult
    vlm_used: bool
    vlm_bbox: Optional[Tuple[int, int, int, int]] = None
    vlm_bbox_used: Optional[Tuple[int, int, int, int]] = None
    vlm_label: Optional[str] = None
    vlm_roi_json_path: Optional[Path] = None
    kept_instance_ids: List[int] = field(default_factory=list)
    source_instance_indices: List[int] = field(default_factory=list)
    intersection_pixels: Dict[int, int] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)


def detect_vlm_roi(
    rgb_path: Path,
    *,
    prompt: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_VLM_TIMEOUT_S,
    temperature: Optional[float] = None,
) -> Optional[VlmRoiResult]:
    """调用 VLM 返回单个 ROI；无目标或解析失败返回 ``None``。"""
    rgb_path = rgb_path.expanduser().resolve()
    if not rgb_path.is_file():
        raise FileNotFoundError(f"rgb not found: {rgb_path}")

    width, height = _read_image_size(rgb_path)
    prompt_text = prompt if prompt is not None else _vlm_prompt()
    url = api_url if api_url is not None else _vlm_api_url()
    model_name = model if model is not None else _vlm_model()
    temp = DEFAULT_VLM_TEMPERATURE if temperature is None else float(temperature)

    image_base64 = _image_to_base64(rgb_path)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": temp,
    }

    print(f"[vlm_seg] POST {url} model={model_name!r}")
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"VLM 服务不可达: {url}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"VLM request failed: status={resp.status_code} body={resp.text[:500]}")

    generated_text = resp.json()["choices"][0]["message"]["content"]
    det = _parse_vlm_detections(generated_text)
    if det is None:
        print(f"[vlm_seg] VLM parse failed or empty: {generated_text[:200]!r}")
        return None

    bbox_norm = [float(x) for x in det["bbox_2d"]]
    x1, y1, x2, y2 = bbox_to_pixel(bbox_norm, width, height)
    if (x2 - x1 + 1) * (y2 - y1 + 1) < _vlm_roi_min_area():
        print(f"[vlm_seg] ROI too small: {(x1, y1, x2, y2)}")
        return None

    label = str(det.get("label", "roi"))
    print(f"[vlm_seg] ROI pixel=({x1},{y1},{x2},{y2}) label={label!r}")
    return VlmRoiResult(
        bbox_pixel=(x1, y1, x2, y2),
        label=label,
        raw_response=generated_text,
        bbox_norm=bbox_norm,
    )


def detect_vlm_detections(
    rgb_path: Path,
    *,
    prompt: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_VLM_TIMEOUT_S,
    temperature: Optional[float] = None,
    max_tokens: int = 512,
) -> Optional[VlmDetectionsResult]:
    """调用 VLM 并解析全部检测框；无有效框时返回 ``None``。"""
    rgb_path = rgb_path.expanduser().resolve()
    if not rgb_path.is_file():
        raise FileNotFoundError(f"rgb not found: {rgb_path}")

    width, height = _read_image_size(rgb_path)
    prompt_text = prompt if prompt is not None else _vlm_prompt()
    url = api_url if api_url is not None else _vlm_api_url()
    model_name = model if model is not None else _vlm_model()
    temp = DEFAULT_VLM_TEMPERATURE if temperature is None else float(temperature)

    image_base64 = _image_to_base64(rgb_path)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": max(32, int(max_tokens)),
        "temperature": temp,
    }

    print(f"[vlm_seg] POST {url} model={model_name!r}")
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"VLM 服务不可达: {url}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"VLM request failed: status={resp.status_code} body={resp.text[:500]}")

    generated_text = resp.json()["choices"][0]["message"]["content"]
    raw_dets = _parse_vlm_detection_list(generated_text)
    if not raw_dets:
        print(f"[vlm_seg] VLM parse failed or empty: {generated_text[:200]!r}")
        return None

    items: List[VlmDetectionItem] = []
    for det in raw_dets:
        bbox_norm = [float(x) for x in det["bbox_2d"]]
        x1, y1, x2, y2 = bbox_to_pixel(bbox_norm, width, height)
        label = str(det.get("label", "object"))
        items.append(
            VlmDetectionItem(
                bbox_pixel=(x1, y1, x2, y2),
                label=label,
                bbox_norm=bbox_norm,
            )
        )
        print(f"[vlm_seg] det pixel=({x1},{y1},{x2},{y2}) label={label!r}")

    return VlmDetectionsResult(items=items, raw_response=generated_text)


def _request_vlm_text(
    rgb_path: Path,
    *,
    prompt: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_VLM_TIMEOUT_S,
    temperature: Optional[float] = None,
    max_tokens: int = 512,
) -> str:
    rgb_path = rgb_path.expanduser().resolve()
    if not rgb_path.is_file():
        raise FileNotFoundError(f"rgb not found: {rgb_path}")

    prompt_text = prompt if prompt is not None else _vlm_prompt()
    url = api_url if api_url is not None else _vlm_api_url()
    model_name = model if model is not None else _vlm_model()
    temp = DEFAULT_VLM_TEMPERATURE if temperature is None else float(temperature)

    image_base64 = _image_to_base64(rgb_path)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": max(32, int(max_tokens)),
        "temperature": temp,
    }

    print(f"[vlm_seg] POST {url} model={model_name!r}")
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"VLM 服务不可达: {url}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"VLM request failed: status={resp.status_code} body={resp.text[:500]}")

    return resp.json()["choices"][0]["message"]["content"]


def detect_vlm_points(
    rgb_path: Path,
    *,
    prompt: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_VLM_TIMEOUT_S,
    temperature: Optional[float] = None,
    max_tokens: int = 512,
) -> Optional[VlmPointsResult]:
    """调用 VLM 并解析单个 point_2d；无有效点时返回 ``None``。"""
    rgb_path = rgb_path.expanduser().resolve()
    width, height = _read_image_size(rgb_path)

    generated_text = _request_vlm_text(
        rgb_path,
        prompt=prompt,
        api_url=api_url,
        model=model,
        timeout_s=timeout_s,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_points = _parse_vlm_point_list(generated_text)
    if not raw_points:
        print(f"[vlm_seg] VLM point parse failed or empty: {generated_text[:200]!r}")
        return None

    if len(raw_points) > 1:
        print(f"[vlm_seg] VLM returned {len(raw_points)} points, using first only")

    items: List[VlmPointItem] = []
    for item in raw_points[:1]:
        point_norm = [float(x) for x in (item.get("point_2d") or item.get("point"))]
        px, py = point_to_pixel(point_norm, width, height)
        label = str(item.get("label", "point"))
        items.append(
            VlmPointItem(
                point_pixel=(px, py),
                label=label,
                point_norm=point_norm,
            )
        )
        print(f"[vlm_seg] point pixel=({px},{py}) label={label!r}")

    return VlmPointsResult(items=items, raw_response=generated_text)


def _load_draw_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label_color(label: str, label_colors: Optional[Dict[str, str]] = None) -> str:
    colors = label_colors or DEFAULT_VLM_LABEL_COLORS
    if label in colors:
        return colors[label]
    fallback = ("#ff3333", "#3366ff", "#ff8800", "#cc33ff", "#00cccc")
    return fallback[hash(label) % len(fallback)]


def render_vlm_detections_on_image(
    image: Image.Image,
    items: List[VlmDetectionItem],
    *,
    label_colors: Optional[Dict[str, str]] = None,
) -> Image.Image:
    """在 PIL 图像上绘制多个 VLM 检测框，按 label 着色。"""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _load_draw_font()

    for item in items:
        color = _label_color(item.label, label_colors)
        x1, y1, x2, y2 = item.bbox_pixel
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        text_y = max(0, y1 - 22)
        draw.text((x1, text_y), item.label, fill=color, font=font)
    return img


def render_vlm_points_on_image(
    image: Image.Image,
    items: List[VlmPointItem],
    *,
    label_colors: Optional[Dict[str, str]] = None,
    radius: int = 14,
) -> Image.Image:
    """在 PIL 图像上绘制 VLM pointing 结果（十字准星 + 圆环）。"""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _load_draw_font()

    for item in items:
        color = _label_color(item.label, label_colors)
        x, y = item.point_pixel
        r = max(6, int(radius))
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=3)
        draw.line([(x - r * 2, y), (x + r * 2, y)], fill=color, width=2)
        draw.line([(x, y - r * 2), (x, y + r * 2)], fill=color, width=2)
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=color)
        text_x = min(x + r + 6, img.width - 1)
        text_y = max(0, y - r - 4)
        draw.text((text_x, text_y), item.label, fill=color, font=font)
    return img


def render_vlm_roi_on_image(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    *,
    label: str = "",
    margin_px: int = 0,
) -> Image.Image:
    """在 PIL 图像上绘制 VLM ROI 红框，返回新图像（不写盘）。"""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = _load_draw_font()
    width, height = img.size

    if margin_px > 0:
        ex1, ey1, ex2, ey2 = _expand_bbox(bbox, margin_px, width, height)
        draw.rectangle([(ex1, ey1), (ex2, ey2)], outline="#ff8800", width=2)

    x1, y1, x2, y2 = bbox
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
    if label:
        draw.text((x1, max(0, y1 - 22)), label, fill="red", font=font)
    return img


def draw_vlm_roi_vis(
    rgb_path: Path,
    bbox: Tuple[int, int, int, int],
    output_path: Path,
    *,
    label: str = "",
) -> Path:
    img = render_vlm_roi_on_image(
        Image.open(rgb_path),
        bbox,
        label=label,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _write_vlm_roi_json(
    path: Path,
    *,
    roi: Optional[VlmRoiResult],
    roi_bbox_used: Optional[Tuple[int, int, int, int]],
    raw_detection_json: Optional[Path],
    filtered_detection_json: Optional[Path],
    kept_instance_ids: List[int],
    kept_source_indices: List[int],
    intersection_pixels: Dict[int, int],
    raw_intersection_pixels: Dict[int, int],
) -> Path:
    payload: Dict[str, Any] = {
        "use_vlm_roi_filter": roi is not None,
        "vlm_used": roi is not None,
        "min_intersection_px": _vlm_min_intersection_px(),
        "selection_rule": "max_roi_intersection",
        "raw_detection_ism": str(raw_detection_json) if raw_detection_json else None,
        "filtered_detection_ism": str(filtered_detection_json) if filtered_detection_json else None,
        "kept_instance_ids": kept_instance_ids,
        "kept_source_indices": kept_source_indices,
        "intersection_pixels": {str(k): int(v) for k, v in intersection_pixels.items()},
        "raw_intersection_pixels": {str(k): int(v) for k, v in raw_intersection_pixels.items()},
    }
    if roi is not None:
        payload.update(
            {
                "bbox_pixel": list(roi.bbox_pixel),
                "bbox_pixel_with_margin": list(roi_bbox_used) if roi_bbox_used else None,
                "bbox_norm": roi.bbox_norm,
                "label": roi.label,
            }
        )
    return _save_json(path, payload)


def _publish_raw_debug_artifacts(
    rgb_path: Path,
    output_dir: Path,
    raw_sam3: Sam3SegmentationResult,
    *,
    prompt: Optional[str],
) -> None:
    results_dir = output_dir / "results"
    raw_json_path = results_dir / "detection_ism_raw.json"
    _copy_text_file(raw_sam3.detection_ism_path, raw_json_path)

    if raw_sam3.vis_ism_path and raw_sam3.vis_ism_path.is_file():
        _copy_binary_file(raw_sam3.vis_ism_path, results_dir / "vis_ism_raw.png")
    elif raw_sam3.instance_dets:
        visualize_sam3_ism(
            rgb_path,
            raw_sam3.instance_dets,
            results_dir / "vis_ism_raw.png",
            prompt=prompt or DEFAULT_SAM3_PROMPT,
            instance_ids=list(range(1, raw_sam3.num_instances + 1)),
        )


def _publish_passthrough_results(
    rgb_path: Path,
    output_dir: Path,
    sam3_result: Sam3SegmentationResult,
    *,
    prompt: Optional[str],
) -> Sam3SegmentationResult:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    detection_json = results_dir / "detection_ism.json"
    mask_path = results_dir / "mask_instances.png"
    vis_ism = results_dir / "vis_ism.png"

    _copy_text_file(sam3_result.detection_ism_path, detection_json)
    if sam3_result.mask_exr.is_file():
        _copy_image_file(sam3_result.mask_exr, mask_path)
    if sam3_result.vis_ism_path and sam3_result.vis_ism_path.is_file():
        _copy_binary_file(sam3_result.vis_ism_path, vis_ism)
    elif sam3_result.instance_dets:
        visualize_sam3_ism(
            rgb_path,
            sam3_result.instance_dets,
            vis_ism,
            prompt=prompt or DEFAULT_SAM3_PROMPT,
            instance_ids=list(range(1, sam3_result.num_instances + 1)),
        )
    if mask_path.is_file():
        visualize_sam3_mask_exr(rgb_path, mask_path, results_dir / "vis_sam3_seg.png")

    return Sam3SegmentationResult(
        detection_ism_path=detection_json,
        mask_exr=mask_path if mask_path.is_file() else sam3_result.mask_exr,
        score=sam3_result.score,
        num_instances=sam3_result.num_instances,
        instance_scores=sam3_result.instance_scores,
        instance_dets=sam3_result.instance_dets,
        vis_ism_path=vis_ism if vis_ism.is_file() else sam3_result.vis_ism_path,
    )


def filter_instances_by_roi_intersection(
    rgb_path: Path,
    detection_ism_path: Path,
    output_dir: Path,
    roi_bbox: Tuple[int, int, int, int],
    *,
    sam3_prompt: Optional[str] = None,
    min_intersection_px: Optional[int] = None,
    margin_px: Optional[int] = None,
) -> FilteredInstancesResult:
    """
    解码 ``detection_ism.json`` 中每个实例 mask，只保留与 ROI 矩形区域交集最大的实例。
    """
    rgb_path = rgb_path.expanduser().resolve()
    detection_ism_path = detection_ism_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    width, height = _read_image_size(rgb_path)
    raw_dets = _load_detection_list(detection_ism_path)
    raw_json_path = _copy_text_file(detection_ism_path, results_dir / "detection_ism_raw.json")

    margin = _vlm_roi_margin_px() if margin_px is None else max(0, int(margin_px))
    min_px = _vlm_min_intersection_px() if min_intersection_px is None else max(1, int(min_intersection_px))
    roi_bbox_used = _expand_bbox(roi_bbox, margin, width, height)
    x1, y1, x2, y2 = roi_bbox_used

    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[y1 : y2 + 1, x1 : x2 + 1] = True
    decoded_masks = get_instance_bool_masks(raw_dets, (width, height))

    candidates: List[Tuple[int, Dict[str, Any], np.ndarray, int, float]] = []
    raw_intersection_pixels: Dict[int, int] = {}
    for idx, (det, inst_mask) in enumerate(zip(raw_dets, decoded_masks), start=1):
        intersection_px = int(np.count_nonzero(inst_mask & roi_mask))
        raw_intersection_pixels[idx] = intersection_px
        if intersection_px >= min_px:
            score = float(det.get("score", 0.0))
            candidates.append((idx, det, inst_mask, intersection_px, score))

    if not candidates:
        raise RuntimeError("No SAM3 instances intersect VLM ROI")

    # 按 ROI 交集像素数降序；若并列，则按检测分数降序。
    candidates.sort(key=lambda item: (item[3], item[4]), reverse=True)
    source_idx, det, inst_mask, inter_px, score = candidates[0]
    composite = np.zeros((height, width), dtype=np.uint8)
    filtered_dets: List[Dict[str, Any]] = []
    filtered_scores: List[float] = []
    kept_instance_ids: List[int] = []
    kept_source_indices: List[int] = []
    intersection_pixels: Dict[int, int] = {}
    fill = inst_mask & (composite == 0)
    if np.any(fill):
        composite[fill] = np.uint8(1)
        filtered_dets.append(det)
        filtered_scores.append(score)
        kept_instance_ids.append(1)
        kept_source_indices.append(source_idx)
        intersection_pixels[1] = inter_px

    if not filtered_dets:
        raise RuntimeError("No SAM3 instances intersect VLM ROI")

    filtered_json = results_dir / "detection_ism_filtered.json"
    filtered_alias = results_dir / "detection_ism.json"
    filtered_text = json.dumps(filtered_dets, indent=2, ensure_ascii=False)
    filtered_json.write_text(filtered_text, encoding="utf-8")
    if filtered_alias != filtered_json:
        filtered_alias.write_text(filtered_text, encoding="utf-8")

    mask_path = _save_instance_id_mask_png(results_dir / "mask_instances.png", composite)
    vis_ism_path = visualize_sam3_ism(
        rgb_path,
        filtered_dets,
        results_dir / "vis_ism.png",
        prompt=sam3_prompt or DEFAULT_SAM3_PROMPT,
        instance_ids=kept_instance_ids,
    )
    vis_mask_path = visualize_sam3_mask_exr(
        rgb_path,
        mask_path,
        results_dir / "vis_sam3_seg.png",
    )

    sam3_result = Sam3SegmentationResult(
        detection_ism_path=filtered_alias,
        mask_exr=mask_path,
        score=float(filtered_scores[0]),
        num_instances=len(filtered_dets),
        instance_scores=filtered_scores,
        instance_dets=filtered_dets,
        vis_ism_path=vis_ism_path,
    )
    return FilteredInstancesResult(
        sam3=sam3_result,
        raw_detection_json=raw_json_path,
        filtered_detection_json=filtered_json,
        filtered_detection_json_alias=filtered_alias,
        mask_path=mask_path,
        vis_ism_path=vis_ism_path,
        vis_mask_path=vis_mask_path,
        kept_instance_ids=kept_instance_ids,
        kept_source_indices=kept_source_indices,
        intersection_pixels=intersection_pixels,
        raw_intersection_pixels=raw_intersection_pixels,
        roi_bbox_used=roi_bbox_used,
    )


def run_vlm_sam3_filter_pipeline(
    rgb_path: Path,
    output_dir: Path,
    *,
    vlm_prompt: Optional[str] = None,
    sam3_prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    max_instances: int = 0,
    skip_vlm: bool = False,
) -> VlmSam3FilterResult:
    """
    ① 原图跑 SAM3
    ② 原图跑 VLM
    ③ 用 ROI 过滤 SAM3 实例
    """
    rgb_path = rgb_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timing: Dict[str, float] = {}
    prompt_text = sam3_prompt or _env_first(
        "GENPOSE2_SAM3_PROMPT",
        "SAM6D_SAM3_PROMPT",
        default=DEFAULT_SAM3_PROMPT,
    )
    vlm_enabled = not skip_vlm and use_vlm_roi_filter()

    t0 = time.perf_counter()
    raw_sam3 = run_sam3_segmentation(
        rgb_path,
        output_dir,
        prompt=sam3_prompt,
        threshold=threshold,
        mask_threshold=mask_threshold,
        max_instances=max_instances,
    )
    timing["seg_s"] = time.perf_counter() - t0
    timing["sam3_s"] = timing["seg_s"]

    if not vlm_enabled:
        published = _publish_passthrough_results(rgb_path, output_dir, raw_sam3, prompt=prompt_text)
        roi_json_path = _write_vlm_roi_json(
            results_dir / "vlm_roi.json",
            roi=None,
            roi_bbox_used=None,
            raw_detection_json=published.detection_ism_path,
            filtered_detection_json=published.detection_ism_path,
            kept_instance_ids=list(range(1, published.num_instances + 1)),
            kept_source_indices=list(range(1, published.num_instances + 1)),
            intersection_pixels={},
            raw_intersection_pixels={},
        )
        return VlmSam3FilterResult(
            sam3=published,
            raw_sam3=raw_sam3,
            vlm_used=False,
            vlm_roi_json_path=roi_json_path,
            kept_instance_ids=list(range(1, published.num_instances + 1)),
            source_instance_indices=list(range(1, published.num_instances + 1)),
            timing=timing,
        )

    t0 = time.perf_counter()
    roi = detect_vlm_roi(rgb_path, prompt=vlm_prompt)
    timing["vlm_s"] = time.perf_counter() - t0
    if roi is None:
        raise RuntimeError("VLM 未返回有效 ROI（空框或 JSON 解析失败）。")

    draw_vlm_roi_vis(
        rgb_path,
        roi.bbox_pixel,
        results_dir / "vlm_roi_vis.png",
        label=roi.label,
    )
    _publish_raw_debug_artifacts(rgb_path, output_dir, raw_sam3, prompt=prompt_text)

    t0 = time.perf_counter()
    filtered = filter_instances_by_roi_intersection(
        rgb_path,
        raw_sam3.detection_ism_path,
        output_dir,
        roi.bbox_pixel,
        sam3_prompt=prompt_text,
        min_intersection_px=_vlm_min_intersection_px(),
        margin_px=_vlm_roi_margin_px(),
    )
    timing["instance_filter_s"] = time.perf_counter() - t0

    roi_json_path = _write_vlm_roi_json(
        results_dir / "vlm_roi.json",
        roi=roi,
        roi_bbox_used=filtered.roi_bbox_used,
        raw_detection_json=filtered.raw_detection_json,
        filtered_detection_json=filtered.filtered_detection_json,
        kept_instance_ids=filtered.kept_instance_ids,
        kept_source_indices=filtered.kept_source_indices,
        intersection_pixels=filtered.intersection_pixels,
        raw_intersection_pixels=filtered.raw_intersection_pixels,
    )

    return VlmSam3FilterResult(
        sam3=filtered.sam3,
        raw_sam3=raw_sam3,
        vlm_used=True,
        vlm_bbox=roi.bbox_pixel,
        vlm_bbox_used=filtered.roi_bbox_used,
        vlm_label=roi.label,
        vlm_roi_json_path=roi_json_path,
        kept_instance_ids=filtered.kept_instance_ids,
        source_instance_indices=filtered.kept_source_indices,
        intersection_pixels=filtered.intersection_pixels,
        timing=timing,
    )


def run_vlm_sam3_segmentation(
    rgb_path: Path,
    output_dir: Path,
    *,
    vlm_prompt: Optional[str] = None,
    sam3_prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    mask_exr_out: Optional[Path] = None,
    max_instances: int = 0,
    skip_vlm: bool = False,
) -> VlmSam3FilterResult:
    """兼容旧入口名。``mask_exr_out`` 已忽略，最终输出固定写到 ``results/``。"""
    _ = mask_exr_out
    return run_vlm_sam3_filter_pipeline(
        rgb_path,
        output_dir,
        vlm_prompt=vlm_prompt,
        sam3_prompt=sam3_prompt,
        threshold=threshold,
        mask_threshold=mask_threshold,
        max_instances=max_instances,
        skip_vlm=skip_vlm,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 on original rgb + VLM ROI filter")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args()

    out = run_vlm_sam3_filter_pipeline(
        args.image,
        args.output_dir,
        skip_vlm=args.skip_vlm,
    )
    print(
        json.dumps(
            {
                "vlm_used": out.vlm_used,
                "vlm_bbox": out.vlm_bbox,
                "vlm_bbox_used": out.vlm_bbox_used,
                "kept_instance_ids": out.kept_instance_ids,
                "source_instance_indices": out.source_instance_indices,
                "detection_ism": str(out.sam3.detection_ism_path),
                "vlm_roi_json": str(out.vlm_roi_json_path) if out.vlm_roi_json_path else None,
                "timing": out.timing,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
