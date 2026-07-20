# 抓取 位姿估计 Tab：标记位 P1 → 抓取点 q

对应 Gradio 页签 **「抓取 位姿估计」**（入口：`bash start.sh` → `http://<host>:8000/ui`）。

用于在**传感器深度**上做实例分割与标记位几何，选出最靠近标记位的抓取候选点 `q`，并以 `[x, y, z, rx, ry, rz]` 输出夹爪位姿。

---

## REST：抓取点 q

与 UI「抓取 位姿估计」同源：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/infer/grasp" \
  -F "rgb=@rgb.png" -F "depth=@depth.png" -F "camera=@camera.json"
```

返回 `xyzrxryrz` 与 `images.grasp_vis`（base64）。完整说明：[grasp_api.md](grasp_api.md)。

---

## 输入

| 输入 | 说明 |
|------|------|
| RGB | 与深度同分辨率（若不一致会先 resize 到深度尺寸） |
| 传感器深度 | `uint16` PNG（mm） |
| `camera.json` | 含 `cam_K`、可选 `depth_scale` |
| 实例提示词 | SAM3 文本提示（料盘 / 圆盘等） |
| 标记位提示词 | 可选；默认见 `prompt/green_square_marker.txt` |

依赖外部服务：

- SAM3 API：`POST /infer`（见 `config/grasp_config.json` → `sam3.api_url`）

---

## 流程概览

```
RGB + 传感器深度 + camera.json
        │
        ├─► SAM3（实例提示词）──► 实例 mask / bbox / id_map
        │
        ├─► SAM3（标记提示词，可选）──► 标记 mask
        │         │
        │         ▼
        │   四角对角线中心反投影 + 平面姿态 ──► P1（位置 + 旋转）
        │
        └─► 各实例点云（传感器深度反投影）
                  │
                  ▼
            剔除外点 → p_i（相机系 Z 最小 / 最近）
                  │
                  ▼
            以 p_i 为球心、半径 8 mm 内的点 ──► 聚合中心 q_i
                  │
                  ▼
            有 P1 时：按沿 P1 局部 X 的 |((q−P1)·X)| 排序
            只保留 |dx| 最近的 1 个 q ──► 抓取点
                  │
                  ▼
            姿态 = P1 旋转
            输出 xyzrxryrz = [x,y,z,rx,ry,rz]
```

坐标约定：

- 位置 `x,y,z`：**相机系**，单位 **mm**（X 右、Y 下、Z 前）
- 姿态 `rx,ry,rz`：单位 **°**，由旋转矩阵按 **ZYX** 欧拉角换算后写成 `[rx,ry,rz]`（Roll / Pitch / Yaw）

---

## UI 输出说明

### 快速预览

| 面板 | 内容 |
|------|------|
| 传感器深度 | 伪彩色深度 |
| SAM3 实例 mask | 仅半透明 mask |
| SAM3 实例 bbox | 仅框 + id/score |
| 标记位 P1 | mask / 角点 / 坐标轴（PEM 同款） |
| P1 + 最近 q_i | 最终用于抓取的 q（通常仅 1 个） |
| P1 + 最近 p_ix | 各实例「沿 P1-X 最近点」后再取全局最近者（对照用） |

### 左侧抓取 JSON

标签：**抓取点 q · xyzrxryrz（mm / °）**

示例：

```json
{
  "success": true,
  "xyzrxryrz": [12.3, -40.5, 395.0, 1.2, -0.5, 89.7],
  "unit": { "xyz": "mm", "rx_ry_rz": "deg" },
  "euler_convention": "ZYX (rz,ry,rx) → displayed as [x,y,z,rx,ry,rz]",
  "position_mm": [12.3, -40.5, 395.0],
  "rpy_deg": { "rx": 1.2, "ry": -0.5, "rz": 89.7 },
  "meta": {
    "instance_id": 3,
    "p_i_mm": [...],
    "q_i_mm": [...],
    "x_dist_mm": 22.05,
    "role": "gripper_grasp_point"
  }
}
```

### 3D 点云

- 灰色：背景；彩色：各实例
- 黄球 / 短轴：P1
- 彩色球 + RGB 轴：最终 q（姿态取自 P1；已烧进点云以便 Model3D 可见）

详情 JSON 字段：`instance_qi` / `instance_qi_all`（含 `p_i_mm`、`q_i_mm`、`radius_mm` / `num_sphere` 等）。

---

## 关键符号

| 符号 | 含义 |
|------|------|
| **P1** | 绿色方标标记位中心位姿（位置 + 平面姿态） |
| **p_i** | 实例 i 剔除外点后，相机系 **Z 最小**点 |
| **q_i** | 以 **p_i 为球心、半径 8 mm** 内点的均值中心（UI 展示用） |
| **p_ix** | 实例内沿 **P1 局部 X** 距 P1 最近的点（对照路径） |
| **q** | 全部 q_i 中沿 P1-X `|dx|` **最近**的那一个 → 夹爪抓取点 |

---

## 代码入口

| 路径 | 作用 |
|------|------|
| `src/grasp/sam3_tab.py` | Tab UI、推理编排、抓取 JSON |
| `src/grasp/marker.py` | 标记筛选、P1 可视化、2D 点预览 |
| `src/grasp/place_geometry.py` | 标记几何 / 反投影 / 姿态（移植自 PEM） |
| `src/depth/pointcloud.py` | 实例点云、`find_instance_qi_from_pi_sphere`、P1-X 筛选 |
| `src/grasp/sam3.py` | SAM3 客户端、`render_sam3_mask_bbox_previews` |
| `prompt/green_square_marker.txt` | 默认标记提示词 |
| `config/grasp_config.json` | `sam3` / `place` / `pem` 默认配置 |

---

## 与「融合深度 + SAM-6D」页签的关系

| | 抓取 位姿估计 | 融合深度 + SAM-6D |
|--|-----------|-------------------|
| 目标 | 标记位附近抓取点 q | CAD 配准 6D 位姿（PEM） |
| 深度 | **仅传感器深度** | 可选融合深度或传感器深度 |
| 分割预览 | mask / bbox 分开 | ISM mask / bbox 分开 |
| 输出 | `xyzrxryrz`（q + P1 姿态） | SAM-6D `detection_pem` / 点云轴 |

两者可共用同一组 RGB / depth / camera 样例做对照。

相关远端 API 说明见 [sam6d_rest_api.md](sam6d_rest_api.md)、[pem.md](pem.md)。
