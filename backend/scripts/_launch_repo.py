"""Set cwd to repository root and sys.path to backend; run a task module with runpy."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent


def launch_task_py(relative_under_backend: str) -> None:
    """relative_under_backend e.g. app/notebook_runtime/tasks/rag_part2_runner.py"""
    path = _BACKEND / relative_under_backend
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    os.chdir(_REPO)
    runpy.run_path(str(path), run_name="__main__")


def launch_module_main(module: str) -> None:
    """Run app.notebook_runtime.tasks.rag_part1 main via run_module."""
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    os.chdir(_REPO)
    import runpy as rp

    rp.run_module(module, run_name="__main__")
