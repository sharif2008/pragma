"""Project paths, ``backend/.env`` loading, and sys.path setup for notebooks/scripts."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = BACKEND_ROOT / "storage"
ATTACK_OPTIONS_JSON = STORAGE_DIR / "attack_options.json"
AGENTIC_FEATURES_JSON = STORAGE_DIR / "agentic_features.json"


def find_backend_root() -> Path:
    here = Path.cwd().resolve()
    candidates = [
        here,
        here.parent,
        here / "backend",
        here.parent / "backend",
        BACKEND_ROOT,
    ]
    for cand in candidates:
        if (cand / "app" / "main.py").is_file() and (cand / "scripts").is_dir():
            return cand
    raise FileNotFoundError("Could not find backend/ (expected app/main.py + scripts/)")


def ensure_backend_on_sys_path() -> Path:
    root = find_backend_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def load_project_dotenv() -> None:
    ensure_backend_on_sys_path()
    cwd = Path.cwd().resolve()
    for path in (
        cwd / "backend" / ".env",
        cwd / ".env",
        cwd.parent / ".env",
        cwd.parent / "backend" / ".env",
        BACKEND_ROOT / ".env",
    ):
        if path.is_file():
            load_dotenv(path)
            return
