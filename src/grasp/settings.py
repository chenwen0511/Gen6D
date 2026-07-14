"""从 config/grasp_config.json 加载 VLM / SAM3 / SAM-6D 等服务地址与默认参数。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from src.grasp.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def _config_path() -> Path:
    custom = os.environ.get("GRASP_CONFIG") or os.environ.get("VLM_INFER_CONFIG")
    if custom and custom.strip():
        return Path(custom.strip()).expanduser()
    return DEFAULT_CONFIG_PATH


def load_config() -> Dict[str, Any]:
    path = _config_path()
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是 JSON 对象: {path}")
    return data


def _section(name: str) -> Dict[str, Any]:
    cfg = load_config()
    section = cfg.get(name)
    if not isinstance(section, dict):
        return {}
    return section


def _env_str(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _env_float(*keys: str, default: float) -> float:
    text = _env_str(*keys, default="")
    if not text:
        return float(default)
    return float(text)


def _cfg_str(section: Dict[str, Any], key: str, default: str = "") -> str:
    value = section.get(key, default)
    return str(value).strip() if value is not None else default


def _cfg_float(section: Dict[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    return float(value)


_vlm = _section("vlm")
VLM_API_URL = _env_str("GENPOSE2_VLM_API_URL", default=_cfg_str(_vlm, "api_url"))
VLM_MODEL = _env_str("GENPOSE2_VLM_MODEL", default=_cfg_str(_vlm, "model", "qwen3-vl-4b"))
VLM_TEMPERATURE = _env_float("GENPOSE2_VLM_TEMPERATURE", default=_cfg_float(_vlm, "temperature", 0.2))
VLM_TIMEOUT_S = _env_float("GENPOSE2_VLM_TIMEOUT_S", default=_cfg_float(_vlm, "timeout_s", 120.0))

_sam3 = _section("sam3")
SAM3_API_URL = _env_str("GENPOSE2_SAM3_API_URL", "SAM6D_SAM3_API_URL", default=_cfg_str(_sam3, "api_url"))
SAM3_PROMPT = _env_str("GENPOSE2_SAM3_PROMPT", "SAM6D_SAM3_PROMPT", default=_cfg_str(_sam3, "prompt"))
SAM3_THRESHOLD = _env_float("GENPOSE2_SAM3_THRESHOLD", default=_cfg_float(_sam3, "threshold", 0.41))
SAM3_MASK_THRESHOLD = _env_float(
    "GENPOSE2_SAM3_MASK_THRESHOLD",
    default=_cfg_float(_sam3, "mask_threshold", 0.5),
)
SAM3_TIMEOUT_S = _env_float("GENPOSE2_SAM3_TIMEOUT_S", default=_cfg_float(_sam3, "timeout_s", 300.0))

_pem = _section("pem")
_default_cad = str(PROJECT_ROOT / "CAD_model" / "tray_180mm_object_frame_mm.obj")
SAM6D_BACKEND = _env_str("SAM6D_BACKEND", default=_cfg_str(_pem, "backend", "sam6d"))
SAM6D_API_URL = _env_str("SAM6D_API_URL", "GENPOSE2_PEM_API_URL", default=_cfg_str(_pem, "api_url"))
SAM6D_FILE_SERVER_URL = _env_str(
    "SAM6D_FILE_SERVER_URL",
    "GENPOSE2_PEM_FILE_SERVER_URL",
    default=_cfg_str(_pem, "file_server_url"),
)
SAM6D_OUTPUT_ROOT = _env_str(
    "SAM6D_OUTPUT_ROOT",
    "GENPOSE2_PEM_OUTPUT_ROOT",
    default=_cfg_str(_pem, "output_root"),
)
_default_sam6d_out = "/home/mui/projects/smt/SAM-6D/SAM-6D/service_outputs"
SAM6D_SAM_OUTPUT_ROOT = _env_str(
    "SAM6D_SAM_OUTPUT_ROOT",
    default=_cfg_str(_pem, "sam6d_output_root", _default_sam6d_out),
)
SAM6D_SAM_FILE_SERVER_URL = _env_str(
    "SAM6D_SAM_FILE_SERVER_URL",
    default=_cfg_str(_pem, "sam6d_file_server_url"),
)
SAM6D_TIMEOUT_S = _env_float("SAM6D_TIMEOUT_S", "GENPOSE2_PEM_TIMEOUT_S", default=_cfg_float(_pem, "timeout_s", 600.0))
SAM6D_CAD_PATH = _env_str("SAM6D_CAD_PATH", default=_cfg_str(_pem, "cad_path", _default_cad))
SAM6D_SEG_BACKEND = _env_str("SAM6D_SEG_BACKEND", default=_cfg_str(_pem, "seg_backend", "sam3"))
SAM6D_SEGMENTOR_MODEL = _env_str("SAM6D_SEGMENTOR_MODEL", default=_cfg_str(_pem, "segmentor_model", "sam"))
SAM6D_DET_SCORE_THRESH = _env_float(
    "SAM6D_DET_SCORE_THRESH",
    default=_cfg_float(_pem, "det_score_thresh", 0.3),
)
SAM6D_DEPTH_SOURCE = _env_str(
    "SAM6D_DEPTH_SOURCE",
    default=_cfg_str(_pem, "depth_source", "fused"),
)

DEFAULT_VLM_API_URL = VLM_API_URL
DEFAULT_VLM_MODEL = VLM_MODEL
DEFAULT_VLM_TEMPERATURE = VLM_TEMPERATURE
DEFAULT_VLM_TIMEOUT_S = VLM_TIMEOUT_S
DEFAULT_SAM3_API_URL = SAM3_API_URL
DEFAULT_SAM3_PROMPT = SAM3_PROMPT
DEFAULT_SAM3_THRESHOLD = SAM3_THRESHOLD
DEFAULT_SAM3_MASK_THRESHOLD = SAM3_MASK_THRESHOLD
DEFAULT_SAM3_TIMEOUT_S = SAM3_TIMEOUT_S
DEFAULT_SAM6D_API_URL = SAM6D_API_URL
DEFAULT_SAM6D_FILE_SERVER_URL = SAM6D_FILE_SERVER_URL
DEFAULT_SAM6D_OUTPUT_ROOT = SAM6D_OUTPUT_ROOT
DEFAULT_SAM6D_SAM_OUTPUT_ROOT = SAM6D_SAM_OUTPUT_ROOT
DEFAULT_SAM6D_SAM_FILE_SERVER_URL = SAM6D_SAM_FILE_SERVER_URL
DEFAULT_SAM6D_TIMEOUT_S = SAM6D_TIMEOUT_S
DEFAULT_SAM6D_CAD_PATH = SAM6D_CAD_PATH
DEFAULT_SAM6D_SEG_BACKEND = SAM6D_SEG_BACKEND
DEFAULT_SAM6D_SEGMENTOR_MODEL = SAM6D_SEGMENTOR_MODEL
DEFAULT_SAM6D_DET_SCORE_THRESH = SAM6D_DET_SCORE_THRESH
DEFAULT_SAM6D_DEPTH_SOURCE = SAM6D_DEPTH_SOURCE

# 兼容旧 import 名
DEFAULT_PEM_API_URL = SAM6D_API_URL
DEFAULT_PEM_FILE_SERVER_URL = SAM6D_FILE_SERVER_URL
DEFAULT_PEM_OUTPUT_ROOT = SAM6D_OUTPUT_ROOT
DEFAULT_PEM_TIMEOUT_S = SAM6D_TIMEOUT_S
