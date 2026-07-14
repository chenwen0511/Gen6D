"""项目根目录与静态资源路径。"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPT_DIR = PROJECT_ROOT / "prompt"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "grasp_config.json"
