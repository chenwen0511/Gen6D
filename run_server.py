"""Gen6D HTTP REST API server."""

from __future__ import annotations

import argparse
import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

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
from src.grasp.grasp_infer import infer_grasp_from_uploads
from src.grasp.settings import (
    DEFAULT_PLACE_MARKER_PROMPT,
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
)

ROOT_DIR = Path(__file__).resolve().parent


def _pil_to_b64_png(image: Image.Image | None) -> Optional[str]:
    if image is None:
        return None
    return base64.b64encode(image_to_png_bytes(image)).decode("ascii")


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
        description="Gen6D 深度估计与抓取点 REST API",
        version="0.2.0",
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

    async def _infer_grasp_handler(
        rgb: UploadFile,
        depth: UploadFile,
        camera: UploadFile | None,
        camera_json: str | None,
        prompt: str | None,
        marker_prompt: str | None,
        enable_marker_p1: bool,
        sam3_api_url: str | None,
        threshold: float | None,
        mask_threshold: float | None,
        timeout_s: float | None,
        y_band_mm: float,
    ):
        try:
            rgb_image = Image.open(io.BytesIO(await rgb.read())).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid RGB image: {exc}") from exc

        try:
            depth_image = Image.open(io.BytesIO(await depth.read()))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid depth image: {exc}") from exc

        cam_payload: str | None = None
        if camera is not None:
            try:
                cam_payload = (await camera.read()).decode("utf-8")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid camera file: {exc}") from exc
        elif camera_json and camera_json.strip():
            cam_payload = camera_json
        else:
            raise HTTPException(
                status_code=400,
                detail="请上传 camera 文件，或提供 camera_json 表单字段",
            )

        try:
            result = infer_grasp_from_uploads(
                rgb_image,
                depth_image,
                cam_payload,
                prompt=prompt or DEFAULT_SAM3_PROMPT,
                marker_prompt=marker_prompt or DEFAULT_PLACE_MARKER_PROMPT,
                enable_marker_p1=bool(enable_marker_p1),
                api_url=sam3_api_url or DEFAULT_SAM3_API_URL,
                threshold=threshold if threshold is not None else DEFAULT_SAM3_THRESHOLD,
                mask_threshold=(
                    mask_threshold if mask_threshold is not None else DEFAULT_SAM3_MASK_THRESHOLD
                ),
                timeout_s=timeout_s if timeout_s is not None else DEFAULT_SAM3_TIMEOUT_S,
                y_band_mm=float(y_band_mm),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"grasp infer failed: {exc}") from exc

        body = result.to_api_dict()
        body["images"] = {
            "grasp_vis": _pil_to_b64_png(result.grasp_vis),
            "mask_vis": _pil_to_b64_png(result.mask_vis),
            "bbox_vis": _pil_to_b64_png(result.bbox_vis),
            "p1_vis": _pil_to_b64_png(result.p1_vis),
        }
        status = 200 if result.success else 422
        return JSONResponse(body, status_code=status)

    @app.post(
        "/api/v1/infer/grasp",
        summary="抓取点 q（SAM3 分割 Tab 同款）",
        tags=["grasp"],
    )
    async def infer_grasp_v1(
        rgb: Annotated[UploadFile, File(description="RGB 图像")],
        depth: Annotated[UploadFile, File(description="传感器深度 uint16 PNG（mm）")],
        camera: Annotated[
            UploadFile | None, File(description="camera.json（与 camera_json 二选一）")
        ] = None,
        camera_json: Annotated[
            str | None, Form(description="相机内参 JSON 字符串（与 camera 文件二选一）")
        ] = None,
        prompt: Annotated[str | None, Form(description="实例 SAM3 提示词")] = None,
        marker_prompt: Annotated[str | None, Form(description="标记位 SAM3 提示词")] = None,
        enable_marker_p1: Annotated[bool, Form(description="是否识别标记位并计算 P1")] = True,
        sam3_api_url: Annotated[str | None, Form(description="SAM3 API URL")] = None,
        threshold: Annotated[float | None, Form(description="SAM3 threshold")] = None,
        mask_threshold: Annotated[float | None, Form(description="SAM3 mask_threshold")] = None,
        timeout_s: Annotated[float | None, Form(description="SAM3 超时秒")] = None,
        y_band_mm: Annotated[float, Form(description="p_i 的 y±带宽（mm），聚合 q_i")] = 2.0,
    ):
        return await _infer_grasp_handler(
            rgb,
            depth,
            camera,
            camera_json,
            prompt,
            marker_prompt,
            enable_marker_p1,
            sam3_api_url,
            threshold,
            mask_threshold,
            timeout_s,
            y_band_mm,
        )

    @app.post("/infer/grasp", include_in_schema=True, tags=["grasp"])
    async def infer_grasp_alias(
        rgb: Annotated[UploadFile, File()],
        depth: Annotated[UploadFile, File()],
        camera: Annotated[UploadFile | None, File()] = None,
        camera_json: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        marker_prompt: Annotated[str | None, Form()] = None,
        enable_marker_p1: Annotated[bool, Form()] = True,
        sam3_api_url: Annotated[str | None, Form()] = None,
        threshold: Annotated[float | None, Form()] = None,
        mask_threshold: Annotated[float | None, Form()] = None,
        timeout_s: Annotated[float | None, Form()] = None,
        y_band_mm: Annotated[float, Form()] = 2.0,
    ):
        """短别名，等价于 ``/api/v1/infer/grasp``。"""
        return await _infer_grasp_handler(
            rgb,
            depth,
            camera,
            camera_json,
            prompt,
            marker_prompt,
            enable_marker_p1,
            sam3_api_url,
            threshold,
            mask_threshold,
            timeout_s,
            y_band_mm,
        )

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
