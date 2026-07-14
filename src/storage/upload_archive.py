"""Persist user upload sessions for traceability."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SESSION_DIR = Path("/home/ubuntu/stephen/01-code/Gen6D/outputs/sessions")


def create_session_dir(base_dir: Path = DEFAULT_SESSION_DIR) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = base_dir / stamp
    suffix = 1
    while session_dir.exists():
        session_dir = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def save_depth_mm_png(depth_array: np.ndarray, path: Path) -> None:
    depth = depth_array.astype(np.float32)
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    depth_u16 = np.clip(depth, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_u16).save(path)


def save_session_inputs(
    session_dir: Path,
    rgb: Image.Image,
    source: str = "unknown",
    depth_path: Path | str | None = None,
    depth_bytes: bytes | None = None,
    depth_array: np.ndarray | None = None,
    intrinsics_path: Path | str | None = None,
    intrinsics_data: dict | None = None,
    rgb_bytes: bytes | None = None,
    rgb_suffix: str = ".png",
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    if rgb_bytes is not None:
        (session_dir / f"rgb{rgb_suffix}").write_bytes(rgb_bytes)
    else:
        rgb.convert("RGB").save(session_dir / "rgb.png")

    if depth_path is not None:
        shutil.copy2(depth_path, session_dir / "depth.png")
    elif depth_bytes is not None:
        (session_dir / "depth.png").write_bytes(depth_bytes)
    elif depth_array is not None:
        save_depth_mm_png(depth_array, session_dir / "depth.png")

    if intrinsics_path is not None:
        shutil.copy2(intrinsics_path, session_dir / "camera.json")
    elif intrinsics_data is not None:
        with (session_dir / "camera.json").open("w", encoding="utf-8") as f:
            json.dump(intrinsics_data, f, indent=2)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "files": {
            "rgb": f"rgb{rgb_suffix}" if rgb_bytes is not None else "rgb.png",
            "depth": "depth.png" if (depth_path is not None or depth_bytes is not None or depth_array is not None) else None,
            "camera": "camera.json"
            if (intrinsics_path is not None or intrinsics_data is not None)
            else None,
        },
    }
    with (session_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_session_outputs(session_dir: Path, result) -> None:
    result.processed_rgb.save(session_dir / "processed_rgb.png")
    result.pred_depth_vis.save(session_dir / "pred_depth_vis.png")
    result.conf_vis.save(session_dir / "conf_vis.png")
    if result.sensor_depth_vis is not None:
        result.sensor_depth_vis.save(session_dir / "sensor_depth_vis.png")
    if result.fused_depth_vis is not None:
        result.fused_depth_vis.save(session_dir / "fused_depth_vis.png")

    if result.pointcloud_glb is not None:
        shutil.copy2(result.pointcloud_glb, session_dir / "pointcloud.glb")
    if result.pointcloud_ply is not None:
        shutil.copy2(result.pointcloud_ply, session_dir / "pointcloud.ply")

    summary = result.to_summary()
    summary["session_dir"] = str(session_dir)
    with (session_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def archive_session(
    rgb: Image.Image,
    result,
    source: str = "unknown",
    session_dir: Path | None = None,
    base_dir: Path = DEFAULT_SESSION_DIR,
    depth_path: Path | str | None = None,
    depth_bytes: bytes | None = None,
    depth_array: np.ndarray | None = None,
    intrinsics_path: Path | str | None = None,
    intrinsics_data: dict | None = None,
    rgb_bytes: bytes | None = None,
    rgb_suffix: str = ".png",
) -> Path:
    session_dir = session_dir or create_session_dir(base_dir)
    save_session_inputs(
        session_dir=session_dir,
        rgb=rgb,
        source=source,
        depth_path=depth_path,
        depth_bytes=depth_bytes,
        depth_array=depth_array,
        intrinsics_path=intrinsics_path,
        intrinsics_data=intrinsics_data,
        rgb_bytes=rgb_bytes,
        rgb_suffix=rgb_suffix,
    )
    save_session_outputs(session_dir, result)
    return session_dir
