#!/usr/bin/env python3
"""Prepare tray CAD mesh for CAD-based PEM (SAM-6D / FoundationPose).

Input:  CAD_model/tray_180mm_centered_mesh_v2.ply
Output: CAD_model/processed/tray_180mm/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "CAD_model" / "tray_180mm_centered_mesh_v2.ply"
DEFAULT_OUT_DIR = PROJECT_ROOT / "CAD_model" / "processed" / "tray_180mm"


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected Trimesh, got {type(loaded)}")
    return loaded


def _pca_rotation(vertices: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation: smallest-variance axis -> +Y (卷轴/厚度), disk in XZ."""
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)
    thickness = evecs[:, idx[0]].astype(np.float64)
    in_plane_b = evecs[:, idx[1]].astype(np.float64)
    in_plane_a = evecs[:, idx[2]].astype(np.float64)

    target_y = np.array([0.0, 1.0, 0.0])
    if float(np.dot(thickness, target_y)) < 0:
        thickness = -thickness

    x_axis = in_plane_a - np.dot(in_plane_a, thickness) * thickness
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        x_axis = in_plane_b.copy()
    else:
        x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, thickness)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise RuntimeError("degenerate PCA axes for CAD alignment")
    z_axis = z_axis / z_norm
    r = np.stack([x_axis, thickness, z_axis], axis=1)
    if np.linalg.det(r) < 0:
        z_axis = -z_axis
        r = np.stack([x_axis, thickness, z_axis], axis=1)
    return r.T


def _center_bbox(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = mesh.bounds
    return ((bounds[0] + bounds[1]) * 0.5).astype(np.float64)


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh


def _mesh_stats(mesh: trimesh.Trimesh) -> dict:
    ext = (mesh.bounds[1] - mesh.bounds[0]).astype(float)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_min_mm": mesh.bounds[0].round(4).tolist(),
        "bounds_max_mm": mesh.bounds[1].round(4).tolist(),
        "extent_mm": ext.round(4).tolist(),
        "diameter_mm": float(max(ext[0], ext[2])),
        "thickness_mm": float(ext[1]),
        "centroid_mm": mesh.centroid.round(4).tolist(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def process_tray_cad(
    input_path: Path,
    output_dir: Path,
    *,
    sample_points: int = 4096,
    repair: bool = True,
) -> dict:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_mesh(input_path)
    raw_stats = _mesh_stats(raw)

    center = _center_bbox(raw)
    centered = raw.copy()
    centered.apply_translation(-center)

    rot = _pca_rotation(np.asarray(centered.vertices, dtype=np.float64))
    aligned = centered.copy()
    aligned.apply_transform(_se3(rot, np.zeros(3)))

    if repair:
        aligned = _repair_mesh(aligned)

    # Re-center after rotation (numeric drift).
    c2 = _center_bbox(aligned)
    aligned.apply_translation(-c2)

    stats = _mesh_stats(aligned)

    # Exports (mm).
    mesh_mm_ply = output_dir / "tray_180mm_object_frame_mm.ply"
    mesh_mm_obj = output_dir / "tray_180mm_object_frame_mm.obj"
    aligned.export(mesh_mm_ply)
    aligned.export(mesh_mm_obj)

    # Meters for BOP / FoundationPose-style backends.
    mesh_m = aligned.copy()
    mesh_m.apply_scale(0.001)
    mesh_m_ply = output_dir / "tray_180mm_object_frame_m.ply"
    mesh_m_obj = output_dir / "tray_180mm_object_frame_m.obj"
    mesh_m.export(mesh_m_ply)
    mesh_m.export(mesh_m_obj)

    # Uniform surface point cloud for PEM partial-to-partial matching.
    pts, face_idx = trimesh.sample.sample_surface(aligned, sample_points)
    pts = np.asarray(pts, dtype=np.float32)
    normals = np.asarray(aligned.face_normals)[face_idx].astype(np.float32)

    cloud_ply = output_dir / f"tray_180mm_cad_surface_{sample_points}.ply"
    cloud = trimesh.points.PointCloud(vertices=pts)
    cloud.export(cloud_ply)

    np.save(output_dir / f"tray_180mm_cad_surface_{sample_points}.npy", pts)
    np.save(output_dir / f"tray_180mm_cad_normals_{sample_points}.npy", normals)

    meta = {
        "object_id": "tray_180mm",
        "source_file": str(input_path.relative_to(PROJECT_ROOT))
        if input_path.is_relative_to(PROJECT_ROOT)
        else str(input_path),
        "object_frame": {
            "origin": "mesh bounding-box center after alignment",
            "y_axis": "reel thickness / shortest PCA axis",
            "disk_plane": "XZ (diameter ~180 mm in X and Z)",
            "units_mesh_mm": "millimeters",
            "units_mesh_m": "meters (scale 0.001 from mm mesh)",
        },
        "transforms": {
            "translation_center_mm": (-center).round(6).tolist(),
            "rotation_pca_row_major": rot.round(8).tolist(),
            "post_rotation_recenter_mm": (-c2).round(6).tolist(),
        },
        "raw_stats": raw_stats,
        "processed_stats_mm": stats,
        "outputs": {
            "mesh_mm_ply": str(mesh_mm_ply.relative_to(PROJECT_ROOT)),
            "mesh_mm_obj": str(mesh_mm_obj.relative_to(PROJECT_ROOT)),
            "mesh_m_ply": str(mesh_m_ply.relative_to(PROJECT_ROOT)),
            "mesh_m_obj": str(mesh_m_obj.relative_to(PROJECT_ROOT)),
            "surface_points_ply": str(cloud_ply.relative_to(PROJECT_ROOT)),
            "surface_points_npy": str(
                (output_dir / f"tray_180mm_cad_surface_{sample_points}.npy").relative_to(
                    PROJECT_ROOT
                )
            ),
            "surface_normals_npy": str(
                (output_dir / f"tray_180mm_cad_normals_{sample_points}.npy").relative_to(
                    PROJECT_ROOT
                )
            ),
        },
        "pem_defaults": {
            "cad_mesh_mm": str(mesh_mm_ply.relative_to(PROJECT_ROOT)),
            "cad_mesh_m": str(mesh_m_obj.relative_to(PROJECT_ROOT)),
            "cad_points_npy": str(
                (output_dir / f"tray_180mm_cad_surface_{sample_points}.npy").relative_to(
                    PROJECT_ROOT
                )
            ),
            "sample_points": sample_points,
        },
    }

    meta_path = output_dir / "cad_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    meta["meta_path"] = str(meta_path.relative_to(PROJECT_ROOT))
    return meta


def _se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Process tray CAD for PEM")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-points", type=int, default=4096)
    parser.add_argument("--no-repair", action="store_true")
    args = parser.parse_args()

    meta = process_tray_cad(
        args.input,
        args.output_dir,
        sample_points=args.sample_points,
        repair=not args.no_repair,
    )
    print(json.dumps(meta["processed_stats_mm"], indent=2, ensure_ascii=False))
    print(f"meta: {meta['meta_path']}")


if __name__ == "__main__":
    main()
