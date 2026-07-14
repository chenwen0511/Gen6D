"""Batch depth fusion test for all samples under samples/."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.depth.service import (
    DEFAULT_MODEL_DIR,
    DepthService,
    depth_to_colormap,
    load_intrinsics,
    load_sensor_depth,
)
from src.fusion import fuse_sensor_and_estimated

DEFAULT_SAMPLES_ROOT = ROOT / "samples"


@dataclass
class SamplePaths:
    name: str
    dir: Path
    rgb: Path
    depth: Path
    camera: Path


def resolve_depth_path(sample_dir: Path) -> Path | None:
    raw = sample_dir / "depth_raw.png"
    if raw.exists():
        return raw

    depth = sample_dir / "depth.png"
    if not depth.exists():
        return None

    arr = np.array(Image.open(depth))
    if arr.dtype == np.uint16:
        return depth

    return None


def discover_samples(samples_root: Path) -> list[SamplePaths]:
    samples: list[SamplePaths] = []
    if not samples_root.exists():
        return samples

    for sample_dir in sorted(samples_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        rgb = sample_dir / "rgb.png"
        camera = sample_dir / "camera.json"
        depth = resolve_depth_path(sample_dir)
        if not (rgb.exists() and camera.exists() and depth is not None):
            continue

        samples.append(
            SamplePaths(
                name=sample_dir.name,
                dir=sample_dir,
                rgb=rgb,
                depth=depth,
                camera=camera,
            )
        )
    return samples


def load_depth_scale(camera_json: Path) -> float:
    with camera_json.open(encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("depth_scale", 1.0))


def run_sample(
    sample: SamplePaths,
    service: DepthService,
    output_root: Path | None,
) -> dict:
    intrinsics = load_intrinsics(sample.camera)
    depth_scale = load_depth_scale(sample.camera)
    sensor = load_sensor_depth(sample.depth, depth_scale)
    rgb = np.array(Image.open(sample.rgb))

    pred = service.model.inference(
        [str(sample.rgb)],
        intrinsics=intrinsics[None] if intrinsics is not None else None,
    )
    fusion = fuse_sensor_and_estimated(sensor, pred.depth[0], rgb, pred.conf[0])

    hole = ~fusion.valid_mask
    filled_ratio = (
        float((np.isfinite(fusion.fused_depth[hole]) & (fusion.fused_depth[hole] > 0)).mean())
        if hole.any()
        else 1.0
    )
    black_ratio = float(
        np.all(np.array(depth_to_colormap(fusion.fused_depth)) == 0, axis=-1).mean()
    )

    summary = {
        "sample": sample.name,
        "rgb": str(sample.rgb),
        "depth": str(sample.depth),
        "camera": str(sample.camera),
        "depth_scale": depth_scale,
        "fusion_status": fusion.status,
        "fusion_message": fusion.message,
        "sensor_valid_ratio": float(fusion.valid_mask.mean()),
        "hole_fill_ratio": filled_ratio,
        "fused_black_ratio": black_ratio,
        "scale": fusion.scale,
        "shift": fusion.shift,
        "align_inlier_ratio": fusion.align_inlier_ratio,
        "fusion": fusion.to_summary(),
    }

    if output_root is not None:
        out_dir = output_root / sample.name / "fusion_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        depth_to_colormap(pred.depth[0]).save(out_dir / "pred_depth_vis.png")
        depth_to_colormap(sensor).save(out_dir / "sensor_depth_vis.png")
        depth_to_colormap(fusion.fused_depth).save(out_dir / "fused_depth_vis.png")
        depth_to_colormap(pred.conf[0]).save(out_dir / "conf_vis.png")
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        summary["output_dir"] = str(out_dir)

    return summary


def print_table(results: list[dict]) -> None:
    header = f"{'sample':<22} {'status':<22} {'valid%':>7} {'fill%':>7} {'black%':>7} {'scale':>8}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['sample']:<22} "
            f"{row['fusion_status']:<22} "
            f"{row['sensor_valid_ratio'] * 100:6.1f}% "
            f"{row['hole_fill_ratio'] * 100:6.1f}% "
            f"{row['fused_black_ratio'] * 100:6.1f}% "
            f"{row['scale']:8.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test depth fusion on samples/")
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--sample", action="append", default=[], help="Run only named sample(s)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Save fusion outputs under <output-root>/<sample>/fusion_output/",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualizations to <samples-root>/<sample>/fusion_output/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = discover_samples(args.samples_root)
    if args.sample:
        wanted = set(args.sample)
        samples = [s for s in samples if s.name in wanted]
        missing = wanted - {s.name for s in samples}
        if missing:
            print(f"[error] sample(s) not found: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1

    if not samples:
        print(f"[error] no valid samples found under {args.samples_root}", file=sys.stderr)
        return 1

    output_root = args.output_root
    if args.save and output_root is None:
        output_root = args.samples_root

    print(f"[info] found {len(samples)} sample(s)")
    service = DepthService(model_dir=args.model_dir, device=args.device, save_uploads=False)
    service.load_model()

    results: list[dict] = []
    failed = 0
    for sample in samples:
        print(f"[run] {sample.name}  depth={sample.depth.name}")
        try:
            results.append(run_sample(sample, service, output_root))
        except Exception as exc:
            failed += 1
            print(f"[fail] {sample.name}: {exc}", file=sys.stderr)

    print()
    print_table(results)

    if output_root is not None:
        print()
        print(f"[info] outputs saved under {output_root}/<sample>/fusion_output/")

    bad = [
        r
        for r in results
        if r["hole_fill_ratio"] < 1.0 or r["fused_black_ratio"] > 0.0
    ]
    if bad:
        print(f"[warn] {len(bad)} sample(s) have unfilled holes or black pixels")
        failed += len(bad)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
