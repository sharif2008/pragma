#!/usr/bin/env python3
"""Remove pipeline run artifacts (traffic_run_*.json, customer_message_*.txt) from the knowledge base."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services import kb_service


def main() -> int:
    settings = get_settings()
    db = SessionLocal()
    try:
        result = kb_service.purge_pipeline_run_kb_artifacts(db, settings)
    finally:
        db.close()
    print(json.dumps(result, indent=2))
    print(f"deleted_kb={len(result['deleted_kb_public_ids'])} orphan_files={len(result['deleted_orphan_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
