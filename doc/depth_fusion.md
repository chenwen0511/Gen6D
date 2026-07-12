# 深度融合方案（Depth Fusion）

Gen6D Phase 3 核心模块：将 **传感器深度**（物理准确、有空洞）与 **DA3 估计深度**（完整、相对尺度）融合，输出可用于点云重建与 6D 位姿估计的 **度量融合深度图**。

---

## 1. 背景与动机

### 1.1 工业场景痛点

在电子料盘、反光/半透明塑料等场景中：

| 数据源 | 优势 | 致命缺陷 |
|--------|------|----------|
| **传感器深度**（结构光 / ToF） | 毫米级物理尺度，真实可靠 | 反光/透明材质导致大面积「黑洞」，缺失常出现在物体边缘 |
| **深度估计模型**（DA3 等） | 边缘完整、泛化强，可补全黑洞 | 输出多为相对深度，Scale/Shift 不准，无法直接用于机械臂 Z 轴下探 |

### 1.2 融合原则

> **传感器定尺度 + 大模型补残缺**

- 传感器数据 → **物理锚点**（trusted region）
- 估计深度 → **几何填充**（inpaint region）

参考分析：[gemini_instruction.md](gemini_instruction.md)

---

## 2. 在整体 Pipeline 中的位置

```
RGB ──► DA3 深度估计 ──► D_est ──┐
                                  ├──► 深度融合 ──► D_fused ──► 点云 / 位姿估计
RGB ──► 传感器 ──► D_sensor ─────┘
         相机内参 K
```

当前项目进度：
- ✅ DA3 推理（`src/depth/service.py`）
- ✅ 点云可视化（`src/depth/pointcloud.py`）
- ✅ **深度融合（`src/fusion/`）**

---

## 3. 输入与输出

### 3.1 输入

| 名称 | 格式 | 说明 |
|------|------|------|
| `rgb` | H×W×3 uint8 | 与深度对齐的彩色图 |
| `D_sensor` | H×W float32 | 传感器深度，单位 mm（uint16 需除以 `depth_scale`） |
| `D_est` | H×W float32 | DA3 预测深度（相对深度） |
| `K` | 3×3 float32 | 相机内参，`camera.json` 中 `cam_K` |
| `conf_est` | H×W float32（可选） | DA3 置信度，用于过滤低质量估计 |
| `mask_obj` | H×W bool（可选） | 物体/前景掩码，限制融合与对齐区域 |

### 3.2 输出

| 名称 | 说明 |
|------|------|
| `D_fused` | H×W float32，度量深度（mm），与 `D_sensor` 同尺度 |
| `M_valid` | 传感器有效像素掩码 |
| `M_fused` | 融合区域掩码（哪些像素来自估计补全） |
| `s`, `t` | 对齐参数，满足 `D_metric_est = s · D_est + t` |
| 点云（可选） | 由 `D_fused` + RGB + K 反投影，物理尺度正确 |

### 3.3 数据示例

测试数据：`test/5_piece/`、`test/wite_wall/`

```
rgb.png          # 640×480
depth.png        # 640×480 uint16，有效约 77%
camera.json      # cam_K + depth_scale
```

DA3 推理后深度分辨率为处理分辨率（如 504×378），融合前需 **对齐分辨率与内参**。

---

## 4. 算法流程

```
┌─────────────┐   ┌─────────────┐
│  D_sensor   │   │   D_est     │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
  ① 预处理与对齐分辨率（resize / 内参缩放）
       │
       ▼
  ② 生成传感器有效掩码 M_valid
       │
       ▼
  ③ Scale & Shift 对齐 → D_metric_est
       │
       ▼
  ④ 掩码替换融合 → D_fused_raw
       │
       ▼
  ⑤ 引导滤波平滑边界 → D_fused
       │
       ▼
  ⑥ 点云生成 / 下游位姿
```

---

## 5. 各步骤详细设计

### Step 1：预处理与分辨率对齐

**问题**：传感器 640×480，DA3 输出 504×378，尺寸不一致。

**策略**（推荐）：
1. 以 **传感器分辨率** 为统一输出分辨率（保持物理像素与 K 一致）
2. 将 `D_est`、`conf_est` **双线性插值** resize 到传感器尺寸
3. 若 DA3 使用了不同内参，对 `K` 做尺度变换：

```
sx = W_out / W_src,  sy = H_out / H_src
fx' = fx · sx,  cx' = cx · sx
fy' = fy · sy,  cy' = cy · sy
```

**注意**：优先使用 DA3 返回的 `intrinsics`（已适配处理分辨率），再映射回传感器坐标系。

---

### Step 2：传感器有效掩码 M_valid

定义哪些像素参与对齐、哪些区域保留传感器深度：

```python
M_valid = (D_sensor > d_min) & (D_sensor < d_max) & isfinite(D_sensor)
```

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `d_min` | 50 mm | 过滤过近噪点 |
| `d_max` | 5000 mm | 过滤异常远点 |
| 可选 | 中值滤波 3×3 | 去除椒盐噪声 |

可选增强：
- 结合 `conf_est`：仅在 `conf_est > τ` 的估计区域允许补全
- 结合 `mask_obj`：只在前景内融合，减少背景干扰

---

### Step 3：Scale & Shift 对齐

在 `M_valid` 区域内，将相对深度 `D_est` 对齐到传感器度量空间：

```
D_metric_est = s · D_est + t
```

**求解方式**：

| 方法 | 适用 | 说明 |
|------|------|------|
| 最小二乘 | 噪声小、有效点多 | 最小化 Σ(D_sensor - s·D_est - t)² |
| **RANSAC**（推荐） | 工业场景有离群点 | 鲁棒拟合 s、t，抗反光边缘错误深度 |

**RANSAC 流程**：
1. 随机采样 N 对 `(D_est, D_sensor)`
2. 拟合 `s, t`，统计 inlier 数量
3. 取 inlier 最多的模型，用全部 inlier 重新最小二乘精修

**质量检查**：
- inlier 比例 < 30% → 对齐失败，fallback 仅输出传感器深度或报错
- 记录 `s, t` 用于日志与调试

---

### Step 4：掩码替换融合

```
D_fused_raw = where(M_valid, D_sensor, D_metric_est)
```

| 区域 | 策略 |
|------|------|
| `M_valid == True` | **保留** `D_sensor`（物理锚点） |
| `M_valid == False` | **填入** `D_metric_est`（几何补全） |

**边界过渡带**（可选）：
- 对 `M_valid` 腐蚀/膨胀得到边界带 `M_boundary`
- 在边界带内做加权混合，避免硬切换：

```
D = α · D_sensor + (1-α) · D_metric_est,   α ∈ [0,1]
```

---

### Step 5：引导滤波平滑

硬替换会在补全边界产生深度台阶，影响法线估计与 ICP。

**方法**（二选一或串联）：
1. **Guided Filter**：以 RGB 为引导，平滑 `D_fused_raw`，保留边缘
2. **Joint Bilateral Filter**：深度双边滤波，空间+颜色权重

**注意**：
- 仅在 `M_fused`（补全区域）及边界带应用平滑
- `M_valid` 内传感器像素 **保持不变**，避免损失物理精度

推荐参数起点：
- Guided Filter: `radius=8, eps=1e-2`
- 仅对 `~M_valid` 及边界 5px 生效

---

### Step 6：点云生成与验证

使用融合深度反投影（与现有 `pointcloud.py` 对接，输入改为 `D_fused`）：

```
X = (u - cx) · Z / fx
Y = (v - cy) · Z / fy
Z = D_fused
```

**验证指标**（Phase 3.6）：

| 指标 | 方法 |
|------|------|
| 完整性 | 有效像素比例 vs 纯传感器 |
| 尺度准确性 | 已知物体尺寸 / 平面距离误差 |
| 边缘连续性 | 法线方差、深度梯度在边界处 |
| 点云视觉 | UI 对比三种点云：sensor / est / fused |

---

## 6. 模块设计（待实现）

建议目录：

```
src/fusion/
├── __init__.py
├── config.py           # FusionConfig
├── types.py            # FusionResult
├── resize.py           # 分辨率对齐
├── mask.py             # build_valid_mask()
├── align.py            # align_scale_shift()
├── smooth.py           # guided filter / 边界混合
└── pipeline.py         # fuse_sensor_and_estimated()
```

### 6.1 核心接口（草案）

```python
@dataclass
class FusionResult:
    fused_depth: np.ndarray      # H×W, mm
    metric_est_depth: np.ndarray
    valid_mask: np.ndarray
    inpaint_mask: np.ndarray
    scale: float
    shift: float
    align_inlier_ratio: float

def fuse_sensor_and_estimated(
    sensor_depth: np.ndarray,
    est_depth: np.ndarray,
    intrinsics: np.ndarray,
    conf: np.ndarray | None = None,
    obj_mask: np.ndarray | None = None,
) -> FusionResult:
    ...
```

### 6.2 与现有服务集成

```
DepthService.predict()
    ├── DA3 推理 → D_est, conf
    ├── 读取 D_sensor（可选）
    └── fuse_sensor_and_estimated() → D_fused
            └── build_pointcloud_files(D_fused)  # 物理尺度点云
```

UI / API 增加输出：
- 融合深度伪彩色图
- 三种点云对比（sensor / est / fused）或切换视图

---

## 7. 下游位姿估计使用策略

融合深度到位姿模块（FoundationPose / GenPose2）：

### 7.1 初始位姿
- 输入：**融合深度** + RGB + 掩码 + CAD 模型
- 完整几何 → 提高 Render-and-Compare 假设生成成功率

### 7.2 ICP 精细化权重

| 点来源 | 权重 |
|--------|------|
| 传感器有效点（`M_valid`） | 高（如 1.0） |
| 估计补全点（`~M_valid`） | 低（如 0.2～0.5） |

兼顾 **完整性约束** 与 **物理精度**。

---

## 8. 对比实验计划

在 `test/5_piece`、`test/wite_wall` 及更多料盘数据上：

| 实验组 | 深度来源 | 预期 |
|--------|----------|------|
| A | 纯传感器 | 尺度准，有空洞 |
| B | 纯 DA3（对齐后） | 完整，尺度依赖对齐 |
| C | **融合** | 完整 + 物理尺度 |

记录：有效像素率、对齐 s/t、点云可视化、后续位姿误差（Phase 5）。

---

## 9. 实现步骤（Phase 3 Checklist）

- [x] **3.1** 实现 `build_valid_mask()`：传感器有效掩码
- [x] **3.2** 实现 `align_scale_shift()`：最小二乘 + RANSAC
- [x] **3.3** 实现 `fuse_depths()`：掩码替换 + 边界混合
- [x] **3.4** 实现 `guided_smooth()`：引导滤波，仅作用于补全区
- [x] **3.5** 分辨率/内参对齐工具函数
- [x] **3.6** 接入 `DepthService`，UI 展示 fused 深度与点云
- [x] **3.7** 单元测试：对齐精度、边界连续性、空输入 fallback
- [ ] **3.8** 对比实验 A/B/C，记录 baseline

---

## 10. 风险与 Fallback

| 风险 | 处理 |
|------|------|
| 有效传感器点过少（<5%） | 对齐不可靠 → 仅输出 DA3 对齐结果并告警 |
| 全图传感器失效 | fallback 纯估计深度 |
| 对齐 inlier 过低 | 不融合，返回传感器 + 日志 |
| s/t 异常（负 scale 等） | 拒绝对齐，人工检查标定 |

---

## 11. 参考

- 整体 Pipeline：[README.md](../README.md)
- 方案设计图：[pipeline.png](pipeline.png)
- 融合动机分析：[gemini_instruction.md](gemini_instruction.md)
- DA3 使用说明：[da_basic_usage.md](da_basic_usage.md)
- 当前推理服务：`src/depth/service.py`
