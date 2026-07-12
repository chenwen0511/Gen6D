"""Gen6D Gradio UI for quick depth inference demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np

from src.depth.service import DEFAULT_MODEL_DIR, DepthService, load_intrinsics_from_dict


def parse_camera_json_file(file_path: str | None) -> tuple[np.ndarray | None, float]:
    if file_path is None:
        return None, 1.0

    path = Path(file_path)
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        intrinsics = load_intrinsics_from_dict(data)
        depth_scale = float(data.get("depth_scale", 1.0))
        return intrinsics, depth_scale
    except json.JSONDecodeError as exc:
        raise gr.Error(f"内参 JSON 格式错误: {exc}") from exc
    except OSError as exc:
        raise gr.Error(f"无法读取内参文件: {exc}") from exc


def build_ui(service: DepthService) -> gr.Blocks:
    def run_upload(rgb, depth, intrinsics_file):
        if rgb is None:
            raise gr.Error("请上传 RGB 图像")

        intrinsics, depth_scale = parse_camera_json_file(intrinsics_file)

        sensor_depth = None
        if depth is not None:
            arr = depth.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            arr[arr == 0] = np.nan
            sensor_depth = arr / depth_scale

        result = service.predict(rgb, intrinsics=intrinsics, sensor_depth=sensor_depth)
        summary = result.to_summary()
        if result.pointcloud_glb is None:
            summary["pointcloud"] = "未生成（需要上传相机内参，或等待模型输出内参）"
        else:
            summary["pointcloud_glb"] = str(result.pointcloud_glb)
            summary["pointcloud_ply"] = str(result.pointcloud_ply)
        info = json.dumps(summary, indent=2, ensure_ascii=False)

        glb_file = str(result.pointcloud_glb) if result.pointcloud_glb else None
        ply_file = str(result.pointcloud_ply) if result.pointcloud_ply else None
        return (
            result.processed_rgb,
            result.pred_depth_vis,
            result.conf_vis,
            result.sensor_depth_vis,
            glb_file,
            glb_file,
            ply_file,
            info,
        )

    with gr.Blocks(title="Gen6D Depth Demo") as demo:
        gr.Markdown(
            "# Gen6D 深度估计演示\n"
            "上传 RGB 图像（可选传感器深度 + 相机内参文件）进行推理。"
            "点云默认降采样至 5 万点，支持浏览器预览与 GLB/PLY 下载。"
        )

        with gr.Row():
            rgb_input = gr.Image(type="pil", label="RGB 图像", height=320)
            depth_input = gr.Image(type="numpy", label="传感器深度（可选）", height=320)
            intrinsics_input = gr.File(
                label="相机内参 JSON（点云推荐上传）",
                file_types=[".json"],
                type="filepath",
            )

        upload_btn = gr.Button("运行推理", variant="primary", size="lg")

        gr.Markdown("### 推理结果")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**预处理后 RGB**")
                out_rgb = gr.Image(show_label=False, height=360, interactive=False)
            with gr.Column():
                gr.Markdown("**预测深度**")
                out_pred = gr.Image(show_label=False, height=360, interactive=False)
            with gr.Column():
                gr.Markdown("**置信度**")
                out_conf = gr.Image(show_label=False, height=360, interactive=False)
            with gr.Column():
                gr.Markdown("**传感器深度**")
                out_sensor = gr.Image(show_label=False, height=360, interactive=False)

        gr.Markdown("### 3D 点云")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**浏览器预览（GLB）**")
                out_pointcloud_3d = gr.Model3D(show_label=False, height=420)
            with gr.Column(scale=1):
                gr.Markdown("**下载点云文件**")
                out_glb_download = gr.File(label="GLB 下载", interactive=False)
                out_ply_download = gr.File(label="PLY 下载", interactive=False)

        out_info = gr.Textbox(label="推理信息", lines=10)

        upload_btn.click(
            run_upload,
            inputs=[rgb_input, depth_input, intrinsics_input],
            outputs=[
                out_rgb,
                out_pred,
                out_conf,
                out_sensor,
                out_pointcloud_3d,
                out_glb_download,
                out_ply_download,
                out_info,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gen6D Gradio UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = DepthService(model_dir=args.model_dir, device=args.device)
    service.load_model()
    demo = build_ui(service)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
