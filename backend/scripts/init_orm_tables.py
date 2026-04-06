#!/usr/bin/env python3
"""
Create all ORM tables in MySQL (SQLAlchemy metadata.create_all).

Requires DATABASE_URL in backend/.env and existing database (db/init_mysql.sql).

  cd backend
  python scripts/init_orm_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.db.session import init_db  # noqa: E402


def main() -> None:
    init_db()
    print("ORM tables ensured (SQLAlchemy metadata.create_all).")


if __name__ == "__main__":
    main()
