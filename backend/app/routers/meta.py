"""Minimal root metadata (no ReDoc, no /api-list, no extra OpenAPI discovery routes)."""

from __future__ import annotations

from fastapi import FastAPI


def register_meta_routes(app: FastAPI) -> None:
    """Attach ``/`` after all routers are mounted."""

    @app.get("/", include_in_schema=False)
    def root_meta() -> dict[str, str]:
        return {
            "message": app.title,
            "health": "/health",
            "docs": "/docs",
        }
