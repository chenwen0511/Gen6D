"""Depth Anything 3 inference validation for Gen6D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.depth.service import (
    DEFAULT_MODEL_DIR,
    DepthService,
    load_intrinsics,
    load_sensor_depth,
)


DEFAULT_TEST_DIR = Path(__file__).resolve().parents[2] / "test" / "5_piece"
DEFAULT_OUTPUT_DIR = DEFAULT_TEST_DIR / "da3_output"


def run_inference(
    model_dir: Path,
    test_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> None:
    rgb_path = test_dir / "rgb.png"
    depth_path = test_dir / "depth.png"

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading model from {model_dir}")
    service = DepthService(model_dir=model_dir, device=device)
    service.load_model()

    print(f"[2/4] Running inference on {rgb_path}")
    result = service.predict(
        rgb_path,
        intrinsics=load_intrinsics(test_dir / "camera.json"),
        sensor_depth=load_sensor_depth(depth_path),
    )

    print("[3/4] Inference result")
    print(f"  depth shape:      {result.pred_depth.shape}")
    print(f"  depth range:      [{result.depth_range[0]:.4f}, {result.depth_range[1]:.4f}]")
    print(f"  conf range:       [{result.conf_range[0]:.4f}, {result.conf_range[1]:.4f}]")
    if result.sensor_valid_ratio is not None:
        print(f"  sensor depth:     valid_ratio={result.sensor_valid_ratio:.2%}, "
              f"range=[{result.sensor_depth_range[0]:.2f}, {result.sensor_depth_range[1]:.2f}] mm")

    np.save(output_dir / "pred_depth.npy", result.pred_depth)
    np.save(output_dir / "pred_conf.npy", result.pred_conf)
    result.pred_depth_vis.save(output_dir / "pred_depth_vis.png")
    if result.sensor_depth_vis is not None:
        result.sensor_depth_vis.save(output_dir / "sensor_depth_vis.png")
    result.processed_rgb.save(output_dir / "processed_rgb.png")

    summary = {
        "model_dir": str(model_dir),
        "test_dir": str(test_dir),
        "rgb_path": str(rgb_path),
        **result.to_summary(),
        "output_dir": str(output_dir),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[4/4] Saved outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Depth Anything 3 inference")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local DA3 model directory",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Test sample directory containing rgb.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <test-dir>/da3_output)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.test_dir / "da3_output")
    run_inference(
        model_dir=args.model_dir,
        test_dir=args.test_dir,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
