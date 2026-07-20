# 抓取点 REST API：`/api/v1/infer/grasp`

与 Gradio **「抓取 位姿估计」** 页签同逻辑：实例分割 → 标记位 P1 → `q_i`（以 p_i 为球心半径 8mm 聚合）→ 沿 P1-X 取最近的抓取点 `q`。

算法细节见 [sam3_seg_tab.md](sam3_seg_tab.md)。

---

## URL

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | **`/api/v1/infer/grasp`** | 推荐正式路径 |
| `POST` | **`/infer/grasp`** | 短别名，行为相同 |

Base：`http://<host>:8000`（与 UI 同进程，`bash start.sh`）

---

## 请求

`Content-Type: multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `rgb` | file | 是 | RGB 图像 |
| `depth` | file | 是 | 传感器深度 uint16 PNG（mm） |
| `camera` | file | 条件 | `camera.json` 文件（与 `camera_json` 二选一） |
| `camera_json` | string | 条件 | 内参 JSON 字符串（与 `camera` 二选一） |
| `prompt` | string | 否 | 实例 SAM3 提示词（默认读配置） |
| `marker_prompt` | string | 否 | 标记位提示词 |
| `enable_marker_p1` | bool | 否 | 默认 `true` |
| `sam3_api_url` | string | 否 | SAM3 服务 URL |
| `threshold` | float | 否 | SAM3 threshold |
| `mask_threshold` | float | 否 | SAM3 mask_threshold |
| `timeout_s` | float | 否 | SAM3 超时（秒） |
| `radius_mm` | float | 否 | 以 p_i 为球心的聚合半径（mm），默认 `8.0` |
| `y_band_mm` | float | 否 | 兼容旧参数，等同 `radius_mm` |

`camera.json` 示例：

```json
{
  "cam_K": [393.12, 0, 322.45, 0, 392.81, 242.94, 0, 0, 1],
  "depth_scale": 1.0
}
```

---

## 响应

成功 `200`；业务失败（如无 q）`422`，仍可能带可视化图。

```json
{
  "success": true,
  "message": "ok",
  "elapsed_s": 3.21,
  "xyzrxryrz": [12.3, -40.5, 395.0, 1.2, -0.5, 89.7],
  "unit": { "xyz": "mm", "rx_ry_rz": "deg" },
  "grasp_pose": {
    "success": true,
    "xyzrxryrz": [12.3, -40.5, 395.0, 1.2, -0.5, 89.7],
    "position_mm": [12.3, -40.5, 395.0],
    "rpy_deg": { "rx": 1.2, "ry": -0.5, "rz": 89.7 },
    "rotation_matrix": [[...], [...], [...]],
    "meta": { "instance_id": 3, "x_dist_mm": 0.25, "role": "gripper_grasp_point" }
  },
  "num_instances": 4,
  "marker_p1": { "enabled": true, "success": true, "p1": { ... } },
  "instance_qi": [ { "instance_id": 3, "q_i_mm": [...], "p_i_mm": [...], "x_dist_mm": 0.25 } ],
  "instance_qi_all": [ ... ],
  "image_size": [378, 504],
  "images": {
    "grasp_vis": "<base64 PNG>",
    "mask_vis": "<base64 PNG>",
    "bbox_vis": "<base64 PNG>",
    "p1_vis": "<base64 PNG>"
  }
}
```

| 字段 | 说明 |
|------|------|
| `xyzrxryrz` | **抓取点**：`[x,y,z,rx,ry,rz]`，xyz=**mm**，角度=**°** |
| `images.grasp_vis` | P1 + 最终 q 的叠加图（与 UI 页签预览一致） |
| `images.mask_vis` / `bbox_vis` | 实例分割对照图（可选使用） |
| `images.p1_vis` | 标记位几何可视化（若开启） |

---

## curl 示例

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/infer/grasp" \
  -F "rgb=@samples/20260712_134531/rgb.png" \
  -F "depth=@samples/20260712_134531/depth.png" \
  -F "camera=@samples/20260712_134531/camera.json" \
  -F "prompt=Plastic Reel Connected With Tape" \
  -o /tmp/grasp_resp.json

# 或短路径
curl -s -X POST "http://127.0.0.1:8000/infer/grasp" \
  -F "rgb=@rgb.png" -F "depth=@depth.png" -F "camera=@camera.json"
```

提取可视化：

```python
import base64, json
from pathlib import Path
data = json.loads(Path("/tmp/grasp_resp.json").read_text())
print(data["xyzrxryrz"])
Path("grasp_vis.png").write_bytes(base64.b64decode(data["images"]["grasp_vis"]))
```

---

## 实现入口

| 路径 | 说明 |
|------|------|
| `run_server.py` | 注册 `/api/v1/infer/grasp` 与 `/infer/grasp` |
| `src/grasp/grasp_infer.py` | 核心推理 `infer_grasp` / `infer_grasp_from_uploads` |
| `src/grasp/sam3_tab.py` | Gradio Tab（算法同源） |
