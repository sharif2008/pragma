"""Canonical paths for static JSON under ``backend/storage`` (agentic + attack catalogs)."""

from pathlib import Path

# This file lives at backend/app/notebook_runtime/storage_paths.py
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BACKEND_ROOT / "storage"
ATTACK_OPTIONS_JSON = STORAGE_DIR / "attack_options.json"
AGENTIC_FEATURES_JSON = STORAGE_DIR / "agentic_features.json"
