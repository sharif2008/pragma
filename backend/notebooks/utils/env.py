"""Load ``backend/.env`` regardless of notebook/kernel working directory."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_dotenv() -> None:
    cwd = Path.cwd().resolve()
    for path in (
        cwd / "backend" / ".env",
        cwd / ".env",
        cwd.parent / ".env",
        cwd.parent / "backend" / ".env",
    ):
        if path.is_file():
            load_dotenv(path)
            return
