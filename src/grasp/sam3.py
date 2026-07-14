"""
SAM3 text-prompt segmentation backend (REST HTTP).

当前通过 SAM3 在线服务调用，例如::

    curl -X POST http://127.0.0.1:18002/infer \
      -H "Content-Type: application/json" \
      -d '{
        "image_path": "/path/to/rgb.png",
        "prompt": "Plastic Reel",
        "threshold": 0.41,
        "mask_threshold": 0.50,
        "save_vis": true,
        "output_dir": "/path/to/output"
      }'

SAM3 服务会将结果写入 ``{output_dir}/sam6d_results/detection_ism.json``，本模块继续复用
这些落盘结果生成 FoundationPose 需要的实例 mask 与可视化。

默认配置见下方常量：

    DEFAULT_SAM3_API_URL
    DEFAULT_SAM3_PROMPT
    DEFAULT_SAM3_THRESHOLD
    DEFAULT_SAM3_MASK_THRESHOLD
    DEFAULT_SAM3_TIMEOUT_S
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image

_COCOMASK = None


def _save_instance_id_mask(composite: np.ndarray, mask_path: Path) -> Path:
    """保存实例 id 图（像素值 1..N 表示各实例）。"""
    mask_path = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.suffix.lower() == ".exr":
        mask_path = mask_path.with_suffix(".png")
    cv2.imwrite(str(mask_path), composite)
    return mask_path


def _cocomask():
    """延迟加载 pycocotools（genpose2 环境需 ``pip install pycocotools``）。"""
    global _COCOMASK
    if _COCOMASK is None:
        try:
            from pycocotools import mask as cocomask_mod
        except ImportError as exc:
            raise ImportError(
                "SAM3 分割依赖 pycocotools，请在当前 Python 环境安装：pip install pycocotools"
            ) from exc
        _COCOMASK = cocomask_mod
    return _COCOMASK

from src.grasp.settings import (
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
)


@dataclass
class Sam3SegmentationResult:
    """SAM3 分割输出：detection_ism.json + GenPose2 用 mask.exr。"""

    detection_ism_path: Path
    mask_exr: Path
    score: float
    num_instances: int = 1
    instance_scores: Optional[List[float]] = None
    instance_dets: Optional[List[Dict[str, Any]]] = None
    vis_ism_path: Optional[Path] = None


# 与 SAM3 可视化颜色约定一致（OpenCV BGR）
_VIS_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 128),
    (64, 64, 255),
    (255, 64, 64),
)


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, object]:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = _cocomask().encode(mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _bbox_xywh_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("SAM3 mask is empty")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]


def _load_mask_image(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    gray = np.array(Image.open(path).convert("L"))
    if gray.shape[0] != height or gray.shape[1] != width:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
    return gray > 127


def _decode_rle_dict(rle: Dict[str, Any]) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    size = rle["size"]
    h, w = int(size[0]), int(size[1])
    decoded = _cocomask().decode({"counts": counts, "size": [h, w]})
    return decoded.astype(bool)


def _pick_from_pred_json(
    data: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, float, List[int]]:
    scores = data.get("pred_scores") or data.get("scores")
    masks = data.get("pred_masks") or data.get("masks")
    boxes = data.get("pred_boxes") or data.get("bbox") or data.get("bboxes")

    if masks is None and "segmentation" in data:
        seg = data["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            mask = _decode_rle_dict(seg)
            score = float(data.get("score", data.get("confidence", 1.0)))
            if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    image_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            return mask, score, _bbox_xywh_from_mask(mask)

    if isinstance(masks, list) and masks:
        idx = 0
        if scores is not None and len(scores):
            idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
            score = float(scores[idx])
        else:
            score = float(data.get("confidence", 1.0))
        item = masks[idx]
        if isinstance(item, dict) and "counts" in item:
            mask = _decode_rle_dict(item)
        elif isinstance(item, (list, np.ndarray)):
            mask = np.asarray(item, dtype=bool)
        else:
            raise RuntimeError(f"unsupported mask entry type in SAM3 json: {type(item)}")
        if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                image_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if boxes is not None and len(boxes) > idx:
            box = boxes[idx]
            if isinstance(box, (list, tuple)) and len(box) == 4:
                if all(0.0 <= float(v) <= 1.0 for v in box):
                    w, h = image_size
                    x, y, bw, bh = [float(v) for v in box]
                    bbox = [int(x * w), int(y * h), int(bw * w), int(bh * h)]
                else:
                    bbox = [int(round(v)) for v in box]
            else:
                bbox = _bbox_xywh_from_mask(mask)
        else:
            bbox = _bbox_xywh_from_mask(mask)
        return mask, score, bbox

    raise RuntimeError("SAM3 json has no usable mask fields")


def _parse_sam3_output(work_dir: Path, rgb_path: Path) -> Tuple[np.ndarray, float, List[int]]:
    with Image.open(rgb_path) as im:
        image_size = im.size  # (W, H)

    for jf in sorted(work_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            try:
                return _pick_from_pred_json(data, image_size)
            except RuntimeError:
                continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "segmentation" in data[0] or "bbox" in data[0]:
                det = max(data, key=lambda d: float(d.get("score", 0.0)))
                seg = det["segmentation"]
                mask = _decode_rle_dict(seg) if isinstance(seg, dict) else np.asarray(seg, dtype=bool)
                if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        image_size,
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                return mask, float(det.get("score", 1.0)), list(det.get("bbox", _bbox_xywh_from_mask(mask)))

    mask_candidates = [
        p
        for p in work_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and "mask" in p.name.lower()
    ]
    if not mask_candidates:
        mask_candidates = list(work_dir.rglob("*.png"))
    if mask_candidates:
        path = sorted(mask_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        mask = _load_mask_image(path, image_size)
        return mask, 1.0, _bbox_xywh_from_mask(mask)

    raise RuntimeError(f"no mask or json found under SAM3 output dir: {work_dir}")


def _sam3_api_url() -> str:
    return DEFAULT_SAM3_API_URL


def _sam3_health_url() -> str:
    parsed = urlparse(_sam3_api_url())
    path = parsed.path or "/infer"
    if path.endswith("/infer"):
        path = path[: -len("/infer")] + "/health"
    else:
        path = "/health"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def _sam3_timeout_s() -> float:
    return DEFAULT_SAM3_TIMEOUT_S


def _validate_sam3_service_config(api_url: str) -> None:
    parsed = urlparse(api_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"SAM3 API URL 无效: {api_url!r}\n"
            f"请检查 DEFAULT_SAM3_API_URL（当前默认 {DEFAULT_SAM3_API_URL}）"
        )


def check_sam3_service_health(*, timeout_s: float = 2.0) -> Dict[str, Any]:
    api_url = _sam3_api_url()
    health_url = _sam3_health_url()
    _validate_sam3_service_config(api_url)
    _validate_sam3_service_config(health_url)
    try:
        resp = requests.get(health_url, timeout=timeout_s)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "api_url": api_url,
            "health_url": health_url,
            "error": str(exc),
        }

    payload: Dict[str, Any] = {
        "ok": resp.ok,
        "api_url": api_url,
        "health_url": health_url,
        "status_code": resp.status_code,
    }
    try:
        payload["response"] = resp.json()
    except ValueError:
        text = resp.text.strip()
        payload["response_text"] = text[:500] if text else ""
    return payload


def _decode_detection_mask(det: Dict[str, Any], image_size: Tuple[int, int]) -> np.ndarray:
    """从 detection_ism 单条记录解码 bool mask，image_size 为 (W, H)。"""
    width, height = image_size
    if "segmentation" in det:
        seg = det["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            mask = _decode_rle_dict(seg)
        else:
            mask = np.asarray(seg, dtype=bool)
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask
    exr = det.get("segmentation_mask_exr")
    if exr and Path(exr).is_file():
        gray = np.array(Image.open(exr).convert("L"))
        if gray.shape[0] != height or gray.shape[1] != width:
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        return gray > 127
    raise RuntimeError("detection entry has no segmentation or segmentation_mask_exr")


def write_genpose_mask_exr_from_ism(
    ism_json_path: Path,
    rgb_path: Path,
    mask_exr_path: Path,
    *,
    max_instances: int = 0,
) -> tuple[float, int, List[float], List[Dict[str, Any]]]:
    """
    读取 ``detection_ism.json``（COCO RLE），写出 GenPose2 ``mask.exr``。

    ``max_instances <= 0`` 表示保留 ISM 中全部实例（最多 254 个 id）；``max_instances=N`` 则按
    score 取 Top-N。mask 像素 id 与 score 排序一致：id=1 为最高分实例。

    :return: (best_score, num_instances, scores_for_mask_id_1..N, dets_for_each_mask_id)
    """
    ism_json_path = Path(ism_json_path).expanduser().resolve()
    rgb_path = Path(rgb_path).expanduser().resolve()
    mask_exr_path = Path(mask_exr_path).expanduser().resolve()

    dets = json.loads(ism_json_path.read_text(encoding="utf-8"))
    if not isinstance(dets, list) or not dets:
        raise RuntimeError(f"empty or invalid detection_ism.json: {ism_json_path}")

    with Image.open(rgb_path) as im:
        image_size = im.size  # (W, H)
    width, height = image_size

    ordered = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    if max_instances <= 0:
        ordered = ordered[:254]
    else:
        ordered = ordered[: max(1, min(max_instances, 254))]

    composite = np.zeros((height, width), dtype=np.uint8)
    best_score = float(ordered[0].get("score", 1.0))
    instance_scores: List[float] = []
    instance_dets: List[Dict[str, Any]] = []
    next_id = 1
    for det in ordered:
        mask = _decode_detection_mask(det, image_size)
        fill = mask & (composite == 0)
        if not np.any(fill):
            continue
        composite[fill] = np.uint8(next_id)
        instance_scores.append(float(det.get("score", 0.0)))
        instance_dets.append(det)
        next_id += 1

    if not np.any(composite > 0):
        raise RuntimeError(f"no foreground pixels in mask from {ism_json_path}")

    num_instances = len(instance_scores)
    saved_mask_path = _save_instance_id_mask(composite, mask_exr_path)
    print(
        f"[sam3_seg] instance mask -> {saved_mask_path} "
        f"(instances={num_instances}/{len(dets)} ism dets, best_score={best_score:.4f})"
    )
    return best_score, num_instances, instance_scores, instance_dets


def build_instance_id_mask_from_detections(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """将 detection 列表合成为实例 id mask（像素 1..N）。"""
    width, height = image_size
    composite = np.zeros((height, width), dtype=np.uint8)
    ordered = sorted(detections, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    next_id = 1
    for det in ordered[:254]:
        mask = _decode_detection_mask(det, image_size)
        fill = mask & (composite == 0)
        if not np.any(fill):
            continue
        composite[fill] = np.uint8(next_id)
        next_id += 1
    if not np.any(composite > 0):
        raise RuntimeError("no foreground pixels in SAM3 detections")
    return composite


def save_instance_mask_from_detections(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    mask_path: Path,
) -> Path:
    """写出 GenPose 可用的实例 id mask（PNG，像素 1..N）。"""
    composite = build_instance_id_mask_from_detections(detections, image_size)
    return _save_instance_id_mask(composite, mask_path)


def visualize_sam3_ism(
    rgb_path: Path,
    instance_dets: List[Dict[str, Any]],
    output_path: Path,
    *,
    prompt: Optional[str] = None,
    mask_alpha: float = 0.5,
    instance_ids: Optional[List[int]] = None,
) -> Path:
    """
    在 RGB 上绘制 SAM3 多实例 mask、bbox 与 score，写入 ``vis_ism.png`` 等路径。

    ``instance_ids`` 与 ``mask.exr`` 中 id 一致时（默认 1..N），图例显示 ``id=#``。
    """
    rgb_path = Path(rgb_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb for SAM3 vis: {rgb_path}")
    height, width = bgr.shape[:2]
    image_size = (width, height)

    if not instance_dets:
        cv2.imwrite(str(output_path), bgr)
        print(f"[sam3_seg_backend] vis (empty dets) -> {output_path}")
        return output_path

    label = (prompt or DEFAULT_SAM3_PROMPT).split()[0] or "obj"
    overlay = bgr.astype(np.float32)

    for idx, det in enumerate(instance_dets):
        mask = _decode_detection_mask(det, image_size)
        color = _VIS_COLORS_BGR[idx % len(_VIS_COLORS_BGR)]
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask] = mask_alpha * color_arr + (1.0 - mask_alpha) * overlay[mask]

        bbox = det.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, bw, bh = [int(round(v)) for v in bbox]
        else:
            x, y, bw, bh = _bbox_xywh_from_mask(mask)
        x2, y2 = x + bw, y + bh

        inst_id = (
            int(instance_ids[idx])
            if instance_ids is not None and idx < len(instance_ids)
            else idx + 1
        )
        score = float(det.get("score", 0.0))
        cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label} id={inst_id} {score:.3f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay.astype(np.uint8))
    print(f"[sam3_seg_backend] vis_ism -> {output_path} ({len(instance_dets)} instances)")
    return output_path


def visualize_sam3_mask_exr(
    rgb_path: Path,
    mask_exr_path: Path,
    output_path: Path,
    *,
    alpha: float = 0.55,
) -> Path:
    """基于实例 id mask 图着色叠加。"""
    rgb_path = Path(rgb_path).expanduser().resolve()
    mask_path = Path(mask_exr_path).expanduser().resolve()
    if mask_path.suffix.lower() == ".exr" and not mask_path.is_file():
        mask_path = mask_path.with_suffix(".png")
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb for SAM3 vis: {rgb_path}")
    id_map = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if id_map is None:
        raise FileNotFoundError(f"cannot read instance mask: {mask_path}")
    if id_map.ndim == 3:
        id_map = id_map[:, :, 0]
    overlay = bgr.astype(np.float32)
    for inst_id in sorted(int(v) for v in np.unique(id_map) if int(v) > 0):
        color = np.array(_VIS_COLORS_BGR[(inst_id - 1) % len(_VIS_COLORS_BGR)], dtype=np.float32)
        region = id_map == inst_id
        overlay[region] = alpha * color + (1.0 - alpha) * overlay[region]
        ys, xs = np.where(region)
        if len(xs):
            cv2.putText(
                overlay,
                f"id={inst_id}",
                (int(xs.min()), max(0, int(ys.min()) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color.tolist(),
                2,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(out), overlay.astype(np.uint8))
    print(f"[sam3_seg] vis_sam3_seg -> {out}")
    return out


def get_instance_bool_masks(
    instance_dets: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> List[np.ndarray]:
    """从 SAM3 detection 列表解码各实例 bool mask，``image_size`` 为 (W, H)。"""
    return [_decode_detection_mask(det, image_size) for det in instance_dets]


def _detection_hit_by_point(
    det: Dict[str, Any],
    image_size: Tuple[int, int],
    px: int,
    py: int,
) -> Tuple[bool, str]:
    """判断 point 是否落在该实例的分割 mask 上（不使用 bbox 回退）。"""
    width, height = image_size
    if not (0 <= px < width and 0 <= py < height):
        return False, ""
    mask = _decode_detection_mask(det, image_size)
    if mask[py, px]:
        return True, "mask"
    return False, ""


def filter_detections_by_point(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    point: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    只保留 point 落在分割 mask 上的实例（多个候选时取 score 最高）。
    """
    filtered, meta = filter_detections_by_points(detections, image_size, [point])
    if meta.get("num_after", 0) == 0:
        meta["point_pixel"] = [int(point[0]), int(point[1])]
        return [], meta
    if meta["num_after"] > 1:
        best = max(
            zip(filtered, meta.get("hits", [])),
            key=lambda pair: float(pair[0].get("score", 0.0)),
        )
        filtered = [best[0]]
        meta["hits"] = [best[1]]
        meta["num_after"] = 1
    meta["point_pixel"] = [int(point[0]), int(point[1])]
    if meta.get("hits"):
        hit = meta["hits"][0]
        meta["filter_hit"] = {
            "source_index": hit.get("source_index"),
            "score": hit.get("score"),
        }
        meta["hit_type"] = hit.get("hit_type")
    return filtered, meta


def filter_detections_by_points(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    points: List[Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    只保留 point 落在分割 mask 上的实例（每个实例最多保留一次）。
    """
    points_pixel = [[int(x), int(y)] for x, y in points]
    meta: Dict[str, Any] = {
        "num_before": len(detections),
        "points_pixel": points_pixel,
        "hits": [],
    }

    kept: List[Dict[str, Any]] = []
    kept_indices: set[int] = set()

    for px, py in points_pixel:
        for idx, det in enumerate(detections):
            if idx in kept_indices:
                continue
            hit, hit_type = _detection_hit_by_point(det, image_size, px, py)
            if not hit:
                continue
            kept_indices.add(idx)
            kept.append(det)
            meta["hits"].append(
                {
                    "point_pixel": [px, py],
                    "source_index": idx,
                    "score": float(det.get("score", 0.0)),
                    "hit_type": hit_type,
                }
            )

    meta["num_after"] = len(kept)
    if not kept:
        meta["filter_hit"] = None
        meta["hit_type"] = None
    return kept, meta


def _close_kernel_for_mask(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (9, 9)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    k = max(5, min(31, int(min(bw, bh) * 0.08)))
    return (k, k)


def _connected_component_at_point_or_largest(
    mask_uint8: np.ndarray,
    px: int,
    py: int,
) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(mask_uint8)
    if num_labels <= 1:
        return mask_uint8.astype(bool)

    height, width = mask_uint8.shape
    if 0 <= px < width and 0 <= py < height and labels[py, px] > 0:
        return labels == labels[py, px]

    best_label = 1
    best_area = 0
    for label_id in range(1, num_labels):
        area = int(np.count_nonzero(labels == label_id))
        if area > best_area:
            best_area = area
            best_label = label_id
    return labels == best_label


def _detection_from_mask(
    mask: np.ndarray,
    score: float,
    template: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    det: Dict[str, Any] = {
        "bbox": _bbox_xywh_from_mask(mask),
        "score": float(score),
        "segmentation": _mask_to_rle(mask),
    }
    if template:
        for key in ("label", "category_id", "category_name"):
            if key in template:
                det[key] = template[key]
    return det


def merge_fragment_detections(
    detections: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    *,
    anchor_point: Optional[Tuple[int, int]] = None,
    close_kernel: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    将同一物体的碎片 mask 合并为单个 detection。

    流程：union 全部 mask → 形态学闭运算桥接细缝 → 取含 anchor 的连通域（或最大块）。
    """
    meta: Dict[str, Any] = {
        "merged": False,
        "num_before": len(detections),
        "num_after": len(detections),
    }
    if len(detections) <= 1:
        return detections, meta

    width, height = image_size
    union_mask = np.zeros((height, width), dtype=bool)
    scores: List[float] = []
    for det in detections:
        union_mask |= _decode_detection_mask(det, image_size)
        scores.append(float(det.get("score", 0.0)))

    kernel_size = close_kernel or _close_kernel_for_mask(union_mask)
    union_uint8 = union_mask.astype(np.uint8)
    if kernel_size[0] > 0 and kernel_size[1] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        union_uint8 = cv2.morphologyEx(union_uint8, cv2.MORPH_CLOSE, kernel)

    if anchor_point is not None:
        px, py = int(anchor_point[0]), int(anchor_point[1])
        final_mask = _connected_component_at_point_or_largest(union_uint8, px, py)
    else:
        final_mask = _connected_component_at_point_or_largest(union_uint8, -1, -1)

    if not np.any(final_mask):
        return detections, meta

    merged_det = _detection_from_mask(
        final_mask,
        max(scores),
        template=detections[0],
    )
    meta.update(
        {
            "merged": True,
            "num_after": 1,
            "close_kernel": [kernel_size[0], kernel_size[1]],
            "merge_scores": scores,
            "merged_bbox": merged_det["bbox"],
            "merged_score": merged_det["score"],
        }
    )
    if anchor_point is not None:
        meta["anchor_point"] = [int(anchor_point[0]), int(anchor_point[1])]
    return [merged_det], meta


@dataclass
class Sam3InferApiResult:
    """SAM3 ``POST /infer`` 直接返回结果（适用于 UI 跨机 base64 调用）。"""

    prompt: str
    num_detections: int
    detections: List[Dict[str, Any]]
    elapsed_ms: float
    raw_response: Dict[str, Any]
    point_filter: Optional[Dict[str, Any]] = None


def _pil_image_to_base64(image: Image.Image) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_image_base64(data: str) -> Image.Image:
    import base64
    import io

    text = (data or "").strip()
    if "," in text and text.startswith("data:"):
        text = text.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(text))).convert("RGB")


def render_sam3_detections_on_image(
    image: Image.Image,
    instance_dets: List[Dict[str, Any]],
    *,
    prompt: Optional[str] = None,
    mask_alpha: float = 0.5,
) -> Image.Image:
    """在 PIL 图像上叠加 SAM3 mask 与 bbox，返回新图像。"""
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    image_size = (width, height)

    if not instance_dets:
        return image.convert("RGB")

    label = (prompt or DEFAULT_SAM3_PROMPT).split()[0] or "obj"
    overlay = bgr.astype(np.float32)

    for idx, det in enumerate(instance_dets):
        mask = _decode_detection_mask(det, image_size)
        color = _VIS_COLORS_BGR[idx % len(_VIS_COLORS_BGR)]
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask] = mask_alpha * color_arr + (1.0 - mask_alpha) * overlay[mask]

        bbox = det.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, bw, bh = [int(round(v)) for v in bbox]
        else:
            x, y, bw, bh = _bbox_xywh_from_mask(mask)
        x2, y2 = x + bw, y + bh
        inst_id = idx + 1
        score = float(det.get("score", 0.0))
        cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label} id={inst_id} {score:.3f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    rgb = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def infer_sam3_with_image(
    image: Image.Image,
    *,
    prompt: Optional[str] = None,
    api_url: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    timeout_s: Optional[float] = None,
    return_vis_base64: bool = True,
    points: Optional[List[Tuple[int, int]]] = None,
    point_labels: Optional[List[int]] = None,
    filter_by_point: bool = True,
) -> Tuple[Sam3InferApiResult, Optional[Image.Image]]:
    """
    通过 ``image_base64`` 调用 SAM3 ``POST /infer``，返回解析结果与可视化图。

    可选传入 ``points`` / ``point_labels`` 与文本提示词联合做 text + point grounding。
    优先使用服务端 ``visualization_base64``；若缺失则在本地根据 detections 渲染。
    """
    api_url = api_url or _sam3_api_url()
    _validate_sam3_service_config(api_url)

    prompt_text = prompt if prompt is not None else DEFAULT_SAM3_PROMPT
    thresh = float(threshold if threshold is not None else DEFAULT_SAM3_THRESHOLD)
    mask_thresh = float(mask_threshold if mask_threshold is not None else DEFAULT_SAM3_MASK_THRESHOLD)
    timeout = float(timeout_s if timeout_s is not None else _sam3_timeout_s())

    payload: Dict[str, Any] = {
        "image_base64": _pil_image_to_base64(image),
        "prompt": prompt_text,
        "threshold": thresh,
        "mask_threshold": mask_thresh,
        "save_vis": True,
        "return_vis_base64": bool(return_vis_base64),
    }

    if points:
        payload["points"] = [[int(x), int(y)] for x, y in points]
        if point_labels is None:
            payload["point_labels"] = [1] * len(points)
        else:
            payload["point_labels"] = [int(v) for v in point_labels]

    print(f"[sam3_seg] POST {api_url}")
    print(f"[sam3_seg] payload keys: {list(payload.keys())} prompt={prompt_text!r}")

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"SAM3 HTTP request failed: {exc}") from exc

    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(f"SAM3 invalid JSON response: status={resp.status_code} body={resp.text[:500]}")

    if resp.status_code != 200 or not body.get("ok", False):
        err = body.get("error") if isinstance(body, dict) else resp.text[:500]
        raise RuntimeError(f"SAM3 infer failed: status={resp.status_code} error={err}")

    detections = body.get("detections") or []
    if not isinstance(detections, list):
        raise RuntimeError(f"SAM3 detections must be a list, got {type(detections).__name__}")

    point_filter_meta: Optional[Dict[str, Any]] = None
    filtered_by_point = False
    if points and filter_by_point:
        labels = point_labels if point_labels is not None else [1] * len(points)
        positive_points = [
            (int(x), int(y))
            for (x, y), label in zip(points, labels)
            if int(label) == 1
        ]
        if positive_points:
            width, height = image.size
            filtered, point_filter_meta = filter_detections_by_points(
                detections,
                (width, height),
                positive_points,
            )
            if not filtered:
                raise RuntimeError(
                    f"SAM3 返回 {len(detections)} 个实例，但 points {positive_points} "
                    "未命中任何分割 mask"
                )
            if len(filtered) < len(detections):
                filtered_by_point = True
            detections = filtered
            if len(detections) > 1:
                merged, merge_meta = merge_fragment_detections(
                    detections,
                    (width, height),
                    anchor_point=positive_points[0],
                )
                if merge_meta.get("merged"):
                    detections = merged
                    filtered_by_point = True
                    if point_filter_meta is not None:
                        point_filter_meta["fragment_merge"] = merge_meta

    result = Sam3InferApiResult(
        prompt=str(body.get("prompt", prompt_text)),
        num_detections=len(detections),
        detections=detections,
        elapsed_ms=float(body.get("elapsed_ms", 0.0)),
        raw_response=body,
        point_filter=point_filter_meta,
    )

    vis_image: Optional[Image.Image] = None
    if filtered_by_point or not body.get("visualization_base64"):
        if detections:
            vis_image = render_sam3_detections_on_image(
                image,
                detections,
                prompt=result.prompt,
            )
    else:
        vis_b64 = body.get("visualization_base64")
        if vis_b64:
            try:
                vis_image = _decode_image_base64(str(vis_b64))
            except Exception as exc:
                print(f"[sam3_seg] failed to decode visualization_base64: {exc}")

    if vis_image is None and detections:
        vis_image = render_sam3_detections_on_image(
            image,
            detections,
            prompt=result.prompt,
        )

    return result, vis_image


def decode_sam3_raw_visualization(
    result: Sam3InferApiResult,
    image: Image.Image,
) -> Optional[Image.Image]:
    """解析 SAM3 原始响应中的 ``visualization_base64``，或根据原始 detections 本地渲染。"""
    body = result.raw_response
    vis_b64 = body.get("visualization_base64")
    if vis_b64:
        try:
            return _decode_image_base64(str(vis_b64))
        except Exception as exc:
            print(f"[sam3_seg] failed to decode raw visualization_base64: {exc}")

    raw_dets = body.get("detections") or []
    if isinstance(raw_dets, list) and raw_dets:
        return render_sam3_detections_on_image(
            image,
            raw_dets,
            prompt=result.prompt,
        )
    return None


def run_sam3_segmentation(
    rgb_path: Path,
    output_dir: Path,
    *,
    prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    mask_exr_out: Optional[Path] = None,
    max_instances: int = 0,
) -> Sam3SegmentationResult:
    rgb_path = rgb_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    api_url = _sam3_api_url()
    _validate_sam3_service_config(api_url)

    prompt_text = (
        prompt
        if prompt is not None
        else DEFAULT_SAM3_PROMPT
    )
    thresh = (
        float(threshold)
        if threshold is not None
        else float(DEFAULT_SAM3_THRESHOLD)
    )
    mask_thresh = (
        float(mask_threshold)
        if mask_threshold is not None
        else float(DEFAULT_SAM3_MASK_THRESHOLD)
    )

    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)
    json_path = sam6d_results / "detection_ism.json"

    payload = {
        "image_path": str(rgb_path),
        "prompt": prompt_text,
        "threshold": float(thresh),
        "mask_threshold": float(mask_thresh),
        "save_vis": True,
        "output_dir": str(output_dir),
    }
    print(f"[sam3_seg_backend] POST {api_url}")
    print(f"[sam3_seg_backend] payload: {json.dumps(payload, ensure_ascii=False)}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(api_url, json=payload, timeout=_sam3_timeout_s())
    except requests.RequestException as exc:
        raise RuntimeError(f"SAM3 HTTP request failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[sam3_seg_backend] elapsed_ms={elapsed_ms:.3f} status_code={resp.status_code}")
    if not resp.ok:
        body = resp.text[:4000] if resp.text else ""
        raise RuntimeError(f"SAM3 HTTP infer failed: status={resp.status_code}\n{body}")
    if resp.text:
        print(f"[sam3_seg_backend] response head: {resp.text[:1000]}")

    if not json_path.is_file():
        print(f"[sam3_seg_backend] {json_path} missing, fallback parse under {output_dir}")
        mask, score, bbox_xywh = _parse_sam3_output(output_dir, rgb_path)
        detection = {
            "scene_id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": bbox_xywh,
            "score": float(score),
            "time": 0.0,
            "segmentation": _mask_to_rle(mask),
        }
        json_path.write_text(json.dumps([detection]), encoding="utf-8")
        print(f"[sam3_seg_backend] wrote {json_path} score={score:.4f}")
    else:
        dets = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(dets, list):
            raise RuntimeError(f"SAM3 detection_ism.json must be a list, got {type(dets).__name__}")
        print(f"[sam3_seg_backend] using {json_path} n_detections={len(dets)}")

    if mask_exr_out is None:
        mask_exr_out = sam6d_results / "mask.exr"
    else:
        mask_exr_out = Path(mask_exr_out)

    best_score, num_instances, instance_scores, instance_dets = write_genpose_mask_exr_from_ism(
        json_path,
        rgb_path,
        mask_exr_out,
        max_instances=max_instances,
    )

    vis_ism_path = sam6d_results / "vis_ism.png"
    vis_sam3_seg_path = sam6d_results / "vis_sam3_seg.png"
    instance_ids = list(range(1, num_instances + 1))
    if instance_dets:
        visualize_sam3_ism(
            rgb_path,
            instance_dets,
            vis_ism_path,
            prompt=prompt_text,
            instance_ids=instance_ids,
        )
        visualize_sam3_mask_exr(rgb_path, mask_exr_out, vis_sam3_seg_path)
    elif vis_ism_path.is_file():
        print(f"[sam3_seg_backend] keep existing vis_ism: {vis_ism_path}")
    else:
        print(f"[sam3_seg_backend] no instance_dets for vis, skip")

    return Sam3SegmentationResult(
        detection_ism_path=json_path,
        mask_exr=mask_exr_out,
        score=best_score,
        num_instances=num_instances,
        instance_scores=instance_scores,
        instance_dets=instance_dets,
        vis_ism_path=vis_ism_path if vis_ism_path.is_file() else None,
    )
