"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _formatter() -> logging.Formatter:
    return logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)


def _has_stream_handler(root: logging.Logger) -> bool:
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (
            sys.stdout,
            sys.stderr,
        ):
            return True
    return False


def _has_file_handler(root: logging.Logger, path: Path) -> bool:
    rp = path.resolve()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename).resolve() == rp:
                    return True
            except (OSError, ValueError, AttributeError):
                continue
    return False


def setup_logging(debug: bool = False, *, log_dir: Path | None = None) -> None:
    """
    Configure root logging: always console; optional file under ``log_dir/app.log``.

    Safe to call when Uvicorn already attached handlers — we only add missing ones.
    """
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    fmt = _formatter()

    if not _has_stream_handler(root):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
        if not _has_file_handler(root, log_file):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
