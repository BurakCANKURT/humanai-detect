"""configs/*.yaml dosyalarini ve .env'i okuyan basit yukleyici."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_env() -> None:
    """.env dosyasini (varsa) ortam degiskenlerine yukler."""
    load_dotenv(PROJECT_ROOT / ".env")


def load_yaml(name: str) -> dict[str, Any]:
    """configs/<name>.yaml dosyasini okur, name uzanti olmadan verilir (orn. 'paths')."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key(env_var: str) -> str | None:
    """Bir saglayicinin API key'ini ortam degiskeninden okur."""
    load_env()
    return os.environ.get(env_var)
