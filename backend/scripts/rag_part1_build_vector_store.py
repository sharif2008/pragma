#!/usr/bin/env python3
"""Build RAG vector store (standalone; uses app.notebook_runtime, not .ipynb)."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from _launch_repo import launch_module_main  # noqa: E402

if __name__ == "__main__":
    launch_module_main("app.notebook_runtime.tasks.rag_part1")
