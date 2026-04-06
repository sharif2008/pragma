"""Filesystem helpers under the unified storage root."""

import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile


def safe_filename(name: str) -> str:
    base = Path(name).name
    return base.replace("..", "_")[:255] or "file"


async def save_upload(
    storage_dir: Path,
    upload: UploadFile,
    *,
    prefix: str = "",
) -> tuple[Path, int]:
    storage_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}{uuid.uuid4().hex}_{safe_filename(upload.filename or 'file')}"
    dest = storage_dir / stem
    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            out.write(chunk)
    await upload.close()
    return dest, size


def remove_path(path: str | Path) -> None:
    p = Path(path)
    if p.is_file():
        p.unlink(missing_ok=True)
    elif p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
