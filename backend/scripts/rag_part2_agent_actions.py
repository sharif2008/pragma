#!/usr/bin/env python3
"""RAG Part 2 action plans (standalone runner merged from notebook; cwd = repo root)."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from _launch_repo import launch_task_py  # noqa: E402

if __name__ == "__main__":
    launch_task_py("app/notebook_runtime/tasks/rag_part2_runner.py")
