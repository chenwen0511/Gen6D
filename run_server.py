"""Gen6D HTTP REST API server."""

from __future__ import annotations

import argparse
import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from run_UI import build_ui
from src.depth.service import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DepthService,
    image_to_png_bytes,
    load_intrinsics_from_dict,
    load_sensor_depth_from_image,
)

ROOT_DIR = Path(__file__).resolve().parent


def create_app(model_dir: str, device: str) -> FastAPI:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    service = DepthService(model_dir=model_dir, device=device, output_dir=output_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.load_model()
        yield

    app = FastAPI(
        title="Gen6D API",
        description="Gen6D 深度估计 REST API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.mount("/files/pointclouds", StaticFiles(directory=str(output_dir)), name="pointclouds")

    @app.get("/")
    def root():
        return RedirectResponse(url="/ui")

    @app.post("/api/v1/depth/predict")
    async def predict_depth(
        rgb: Annotated[UploadFile, File(description="RGB 图像")],
        depth: Annotated[UploadFile | None, File(description="传感器深度图（可选）")] = None,
        intrinsics: Annotated[str | None, Form(description="相机内参 JSON，含 cam_K 字段")] = None,
    ):
        try:
            rgb_bytes = await rgb.read()
            rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid RGB image: {exc}") from exc

        rgb_suffix = Path(rgb.filename or "rgb.png").suffix or ".png"

        intrinsics_arr = None
        intrinsics_data = None
        if intrinsics:
            try:
                intrinsics_data = json.loads(intrinsics)
                intrinsics_arr = load_intrinsics_from_dict(intrinsics_data)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid intrinsics JSON: {exc}") from exc

        sensor_depth = None
        depth_scale = 1.0
        depth_bytes = None
        if depth is not None:
            try:
                depth_bytes = await depth.read()
                depth_image = Image.open(io.BytesIO(depth_bytes))
                if intrinsics_data:
                    depth_scale = float(intrinsics_data.get("depth_scale", 1.0))
                sensor_depth = load_sensor_depth_from_image(depth_image, depth_scale)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid depth image: {exc}") from exc

        try:
            result = service.predict(
                rgb_image,
                intrinsics=intrinsics_arr,
                sensor_depth=sensor_depth,
                source="api",
                depth_bytes=depth_bytes,
                intrinsics_data=intrinsics_data,
                rgb_bytes=rgb_bytes,
                rgb_suffix=rgb_suffix,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        response = result.to_summary()
        response["images"] = {
            "processed_rgb": base64.b64encode(image_to_png_bytes(result.processed_rgb)).decode(),
            "pred_depth_vis": base64.b64encode(image_to_png_bytes(result.pred_depth_vis)).decode(),
            "conf_vis": base64.b64encode(image_to_png_bytes(result.conf_vis)).decode(),
        }
        if result.sensor_depth_vis is not None:
            response["images"]["sensor_depth_vis"] = base64.b64encode(
                image_to_png_bytes(result.sensor_depth_vis)
            ).decode()
        if result.fused_depth_vis is not None:
            response["images"]["fused_depth_vis"] = base64.b64encode(
                image_to_png_bytes(result.fused_depth_vis)
            ).decode()
        if result.fusion is not None:
            response["fusion"] = result.fusion
        if result.pointcloud_glb is not None:
            response["pointcloud"] = {
                "point_count": result.point_count,
                "glb_url": f"/files/pointclouds/{result.pointcloud_glb.name}",
                "ply_url": f"/files/pointclouds/{result.pointcloud_ply.name}",
            }
        return JSONResponse(response)

    demo = build_ui(service)
    app = gr.mount_gradio_app(app, demo, path="/ui")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gen6D REST API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.model_dir, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
