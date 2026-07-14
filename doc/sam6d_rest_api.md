# SAM-6D HTTP Service REST API

服务入口：`sam6d_http_service.py`（FastAPI + uvicorn）

默认启动：

```bash
bash start.sh          # 默认 0.0.0.0:8004（见 start.sh）
# 或
python sam6d_http_service.py --host 0.0.0.0 --port 8000
```

---

## 环境变量（启动前）

| 变量 | 必填 | 说明 |
|------|------|------|
| `SAM6D_CAD_PATH` | 是 | CAD 模型路径（`.ply`，单位 mm） |
| `SAM6D_OUTPUT_ROOT` | 否 | 推理输出根目录，默认 `<repo>/service_outputs` |
| `SAM6D_SEG_BACKEND` | 否 | 默认分割后端：`sam6d_ism`、`yolo_seg` 或 `user_mask`，默认 `sam6d_ism` |
| `SAM6D_YOLO_WEIGHTS` | `yolo_seg` 时 | YOLO 权重路径（`.pt` / `.engine`） |
| `SAM6D_YOLO_IMGSZ` | 否 | YOLO 输入尺寸，默认 `640` |
| `SAM6D_YOLO_CONF` | 否 | YOLO 置信度阈值，默认 `0.25` |
| `SAM6D_YOLO_CLASS_ID` | 否 | YOLO 类别 ID，默认 `0` |
| `SAM6D_CUDA_VISIBLE_DEVICES` | 否 | GPU 编号 |
| `SAM6D_ENV_DIR` | 否 | conda 环境路径，默认 `/home/mui/.micromamba/envs/sam6d` |

---

## 通用说明

- **Content-Type**：`/infer` 使用 `multipart/form-data`
- **并发**：全局 `infer_lock`，同一时刻只处理一个 `/infer` 或 `/warmup` 请求
- **CAD 模型**：由环境变量 `SAM6D_CAD_PATH` 指定，请求中不上传
- **日志**：输出到启动终端（stdout），无独立日志文件
- **错误格式**：FastAPI 标准 JSON，`{"detail": "..."}`

---

## GET `/health`

健康检查与服务配置快照。

### 响应 `200`

```json
{
  "status": "ok",
  "root_dir": "/path/to/SAM-6D",
  "env_dir": "/path/to/micromamba/envs/sam6d",
  "cad_path": "/path/to/model.ply",
  "cad_exists": true,
  "output_root": "/path/to/service_outputs",
  "default_seg_backend": "sam6d_ism",
  "yolo_weights": "/path/to/best.engine",
  "yolo_weights_exists": true
}
```

### 示例

```bash
curl http://127.0.0.1:8004/health
```

---

## POST `/warmup`

重新执行启动预热（模板缓存、YOLO 预加载等）。适用于更换 `SAM6D_CAD_PATH` 或 YOLO 权重后，无需重启进程。

与 `/infer` 共用锁，执行期间其他推理请求会等待。

### 请求

无 body。

### 响应 `200`

```json
{
  "status": "ok"
}
```

### 示例

```bash
curl -X POST http://127.0.0.1:8004/warmup
```

---

## POST `/infer`

上传 RGB、深度图、相机内参，执行完整 6D 姿态估计 pipeline：

1. 确保 CAD 模板已渲染（BlenderProc，结果缓存）
2. 实例分割（`sam6d_ism`、`yolo_seg` 或 `user_mask`）
3. 姿态估计（PEM）
4. 返回得分最高的检测结果

### 请求

**Content-Type：** `multipart/form-data`

#### 文件字段（必填）

| 字段 | 类型 | 说明 |
|------|------|------|
| `rgb` | file | RGB 图像（保存为 `rgb.png`） |
| `depth` | file | 深度图（保存为 `depth.png`，单位 mm） |
| `camera` | file | 相机参数 JSON（保存为 `camera.json`） |
| `mask` | file | 二值 mask PNG（**仅** `seg_backend=user_mask` 时必填，保存为 `mask.png`） |

#### `camera.json` 格式

```json
{
  "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
  "depth_scale": 1.0
}
```

- `cam_K`：3×3 内参矩阵，行优先 9 元数组
- `depth_scale`：深度图像素值到 mm 的缩放系数

#### 查询参数（可选）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `segmentor_model` | string | `sam` | ISM 分割模型：`sam` 或 `fastsam`（仅 `seg_backend=sam6d_ism` 时生效） |
| `seg_backend` | string | 环境变量 | 分割后端：`sam6d_ism`、`yolo_seg` 或 `user_mask` |
| `yolo_weights` | string | 环境变量 | YOLO 权重路径（`yolo_seg` 时必填，可覆盖 `SAM6D_YOLO_WEIGHTS`） |
| `yolo_conf` | float | `0.25` | YOLO 置信度阈值 |
| `yolo_imgsz` | int | `640` | YOLO 输入尺寸 |
| `yolo_class_id` | int | `0` | YOLO 目标类别 ID |
| `det_score_thresh` | float | `0.3` | PEM 检测分数阈值（`yolo_seg` 模式下内部固定为 `0.0`） |
| `mask_score` | float | `1.0` | 用户 mask 写入 `detection_ism.json` 的 score（仅 `user_mask`） |

### 响应 `200`

```json
{
  "score": 0.121,
  "xyz_mm": [-61.437, -131.235, 606.296],
  "rotation_euler_zyx_rad": [2.977, 0.816, 1.273],
  "rotation_order": "zyx",
  "pose_convention": "xyz is camera-frame translation in mm; rx, ry, rz are ZYX Euler angles in radians.",
  "xyzrxryrz": [-61.437, -131.235, 606.296, 2.977, 0.816, 1.273],
  "xyzrxryrz_unit": "mm_rad",
  "result_dir": "/path/to/service_outputs/20260712_210000_a1b2c3d4",
  "detection_ism_path": "/path/to/.../detection_ism.json",
  "detection_pem_path": "/path/to/.../detection_pem.json",
  "vis_ism_path": "/path/to/.../vis_ism.png",
  "vis_pem_path": "/path/to/.../vis_pem.png",
  "all_detections": "/path/to/.../detection_pem.json",
  "timing": {
    "upload_s": 0.001,
    "templates_s": 0.05,
    "ism_s": 12.3,
    "yolo_s": null,
    "mask_s": null,
    "pose_s": 3.2,
    "pipeline_s": 15.55,
    "total_s": 15.551
  }
}
```

#### 响应字段说明

| 字段 | 说明 |
|------|------|
| `score` | 最佳检测置信度 |
| `xyz_mm` | 相机坐标系平移 `[x, y, z]`，单位 mm |
| `rotation_euler_zyx_rad` | ZYX 欧拉角 `[rx, ry, rz]`，单位弧度 |
| `xyzrxryrz` | 平移 + 欧拉角合并为 6 维向量 |
| `result_dir` | 本次推理输出目录 |
| `detection_ism_path` | 分割结果 JSON（`yolo_seg` 模式下同样写入此字段名） |
| `detection_pem_path` | PEM 姿态结果 JSON |
| `vis_ism_path` / `vis_pem_path` | 可视化 PNG 路径 |
| `timing` | 各阶段耗时（秒）；`ism_s` / `yolo_s` / `mask_s` 三选一非 null |

### 错误响应

| 状态码 | 场景 |
|--------|------|
| `400` | `segmentor_model` 非法、`camera.json` 格式错误、`user_mask` 缺少 `mask` 文件 |
| `500` | `SAM6D_CAD_PATH` 未设置或不存在、pipeline 执行失败、无检测结果 |

### 示例

#### SAM3 文本分割（Form 字段）

与 `warmup_http_service.py` 一致，控制参数使用 **multipart form 字段**（非 query）：

```bash
curl -X POST "http://127.0.0.1:8004/infer" \
  -F "rgb=@/path/to/rgb.png" \
  -F "depth=@/path/to/depth.png" \
  -F "camera=@/path/to/camera.json" \
  -F "seg_backend=sam3" \
  -F "sam3_prompt=white circular reel" \
  -F "sam3_threshold=0.41" \
  -F "sam3_mask_threshold=0.50" \
  -F "det_score_thresh=0.3"
```

| Form 字段 | 说明 |
|-----------|------|
| `seg_backend` | 固定 `sam3` |
| `sam3_prompt` | 文本提示词 |
| `sam3_threshold` | SAM3 检测阈值 |
| `sam3_mask_threshold` | SAM3 mask 阈值 |
| `det_score_thresh` | PEM 检测分数阈值 |

#### ISM 分割（默认，query 或 form 均可）

```bash
curl -X POST "http://127.0.0.1:8004/infer" \
  -F "rgb=@/path/to/rgb.png" \
  -F "depth=@/path/to/depth.png" \
  -F "camera=@/path/to/camera.json"
```

#### YOLO 分割

```bash
curl -X POST "http://127.0.0.1:8004/infer?seg_backend=yolo_seg&yolo_conf=0.25&yolo_imgsz=640&yolo_class_id=0&det_score_thresh=0.3" \
  -F "rgb=@/path/to/rgb.png" \
  -F "depth=@/path/to/depth.png" \
  -F "camera=@/path/to/camera.json"
```

若已通过环境变量设置 `SAM6D_YOLO_WEIGHTS`，可省略 query 中的 `yolo_weights`。

#### 用户自带 mask（跳过实例分割）

上传二值 PNG mask（非零像素为前景）。若尺寸与 depth 不一致，服务会按 depth 尺寸 nearest-neighbor 缩放。

```bash
curl -X POST "http://127.0.0.1:8004/infer?seg_backend=user_mask&mask_score=1.0&det_score_thresh=0.3" \
  -F "rgb=@/path/to/rgb.png" \
  -F "depth=@/path/to/depth.png" \
  -F "camera=@/path/to/camera.json" \
  -F "mask=@/path/to/mask.png"
```

`warmup_http_service.py` 同样支持，但 `seg_backend`、`mask_score` 等参数使用 **Form 字段** 而非 query：

```bash
curl -X POST "http://127.0.0.1:8001/infer" \
  -F "rgb=@/path/to/rgb.png" \
  -F "depth=@/path/to/depth.png" \
  -F "camera=@/path/to/camera.json" \
  -F "mask=@/path/to/mask.png" \
  -F "seg_backend=user_mask" \
  -F "mask_score=1.0"
```

---

## 输出目录结构

每次 `/infer` 在 `SAM6D_OUTPUT_ROOT/<request_id>/` 下生成：

```
<request_id>/
├── inputs/
│   ├── rgb.png
│   ├── depth.png
│   ├── camera.json
│   └── mask.png        # 仅 user_mask 模式
├── templates/          → 符号链接到模板缓存
└── sam6d_results/
    ├── detection_ism.json
    ├── detection_pem.json
    ├── vis_ism.png
    └── vis_pem.png
```

`request_id` 格式：`YYYYMMDD_HHMMSS_<8位hex>`

模板缓存在：`<SAM6D_OUTPUT_ROOT>/_template_cache/<cad_fingerprint>/templates/`

---

## 分割后端对比

| | `sam6d_ism` | `yolo_seg` | `user_mask` |
|---|-------------|------------|-------------|
| 实现 | 子进程调用 ISM（SAM/FastSAM + DINOv2） | 进程内 YOLO 分割 | 用户上传 mask → `detection_ism.json` |
| 速度 | 较慢 | 较快 | 最快（跳过分割模型） |
| PEM 阈值 | 使用 `det_score_thresh` | 内部固定 `0.0` | 使用 `det_score_thresh` |
| 依赖 | ISM checkpoint | `SAM6D_YOLO_WEIGHTS` | 无额外模型 |
| 额外输入 | — | — | `mask` PNG 文件 |

---

## OpenAPI

服务启动后可访问交互式文档：

- Swagger UI：`http://127.0.0.1:8004/docs`
- ReDoc：`http://127.0.0.1:8004/redoc`
