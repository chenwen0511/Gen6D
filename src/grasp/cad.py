"""Load processed CAD assets for CAD-based PEM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from src.grasp.paths import PROJECT_ROOT

DEFAULT_OBJECT_ID = "tray_180mm"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "CAD_model" / "processed" / "tray_180mm"
DEFAULT_META_PATH = DEFAULT_PROCESSED_DIR / "cad_meta.json"


def load_cad_meta(meta_path: Path | None = None) -> dict[str, Any]:
    path = (meta_path or DEFAULT_META_PATH).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_cad_path(
    key: str = "cad_mesh_mm",
    *,
    meta: dict[str, Any] | None = None,
    meta_path: Path | None = None,
) -> Path:
    data = meta or load_cad_meta(meta_path)
    rel = data.get("pem_defaults", {}).get(key) or data.get("outputs", {}).get(key.replace("cad_", ""))
    if not rel:
        rel = data["outputs"].get("mesh_mm_ply")
    return (PROJECT_ROOT / rel).resolve()


def load_cad_mesh(
    *,
    unit: str = "mm",
    meta_path: Path | None = None,
) -> trimesh.Trimesh:
    meta = load_cad_meta(meta_path)
    key = "cad_mesh_m" if unit == "m" else "cad_mesh_mm"
    path = resolve_cad_path(key, meta=meta)
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh from {path}")
    return loaded


def load_cad_surface_points(
    *,
    meta_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (N,3) surface points in mm and optional (N,3) normals."""
    meta = load_cad_meta(meta_path)
    pts_path = (PROJECT_ROOT / meta["pem_defaults"]["cad_points_npy"]).resolve()
    points = np.load(pts_path)
    normals_path = meta["outputs"].get("surface_normals_npy")
    normals = np.load(PROJECT_ROOT / normals_path) if normals_path else None
    return points, normals


def cad_bounding_radius_mm(*, meta_path: Path | None = None) -> float:
    """CAD 物体坐标系下顶点到原点的最大距离（mm），用于 PEM 球裁剪可视化。"""
    mesh = load_cad_mesh(meta_path=meta_path)
    return float(np.linalg.norm(mesh.vertices, axis=1).max())
