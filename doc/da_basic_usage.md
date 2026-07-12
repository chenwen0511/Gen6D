## 环境安装（gen6d）

本项目使用独立 conda 环境 `gen6d`，从 `genpose2` 克隆，存放在 `/home/ubuntu/stephen/05-venv/gen6d`。

### 1. 创建环境（首次）

```bash
# 从 genpose2 克隆到自定义目录
conda create --prefix /home/ubuntu/stephen/05-venv/gen6d --clone genpose2 -y

# 注册目录，使 conda env list 显示名字 gen6d（只需执行一次）
conda config --append envs_dirs /home/ubuntu/stephen/05-venv
```

验证：

```bash
conda env list
# 应看到：gen6d    /home/ubuntu/stephen/05-venv/gen6d
```

### 2. 激活环境

```bash
conda activate gen6d
```

### 3. 安装 Depth Anything 3 依赖

在 `Depth-Anything-3` 仓库根目录下执行：

```bash
pip install xformers torch>=2 torchvision
pip install -e .                                    # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70  # for gaussian head
pip install -e ".[app]"                               # Gradio, python>=3.10
pip install -e ".[all]"                               # ALL
```

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)
```

```bash
export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
    --use-backend

# CLI video processing with feature visualization
da3 video assets/examples/robot_unitree.mp4 \
    --fps 15 \
    --use-backend \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/robo \
    --export-format glb-feat_vis \
    --feat-vis-fps 15 \
    --process-res-method lower_bound_resize \
    --export-feat "11,21,31"

# CLI auto mode without backend reuse
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_CLI/SOH \
    --model-dir ${MODEL_DIR}

```