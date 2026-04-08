"""FastAPI entrypoint: modular ML + RAG + agentic API."""

from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import ensure_storage_dirs, get_settings
from app.core.logging import setup_logging
from app.db.session import init_db
from app.routers import agent, datasets, files, health, kb, meta, predictions, runs, simulate, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.debug)
    ensure_storage_dirs(settings)
    init_db()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Production-style API for versioned files, CSV ML training (scikit-learn / XGBoost), "
        "batch predictions, FAISS + SentenceTransformers RAG, and LLM-based agentic actions."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "docExpansion": "list",
        "filter": True,
        "tryItOutEnabled": True,
    },
)

_http_fail_logger = logging.getLogger("app.http.fail")
_HTTP_FAIL_LOG_IGNORE_PREFIXES = (
    "/docs",
    "/openapi.json",
    "/health",
)


@app.middleware("http")
async def log_failed_http_requests(request: Request, call_next):
    """
    Log only failed API calls (>= 400) + unhandled exceptions.

    We intentionally keep successful requests quiet to reduce log noise and
    to make "why did the dashboard fail?" investigations faster.
    """
    path = request.url.path
    if any(path.startswith(p) for p in _HTTP_FAIL_LOG_IGNORE_PREFIXES):
        return await call_next(request)

    t0 = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        ms = int((time.perf_counter() - t0) * 1000)
        _http_fail_logger.exception(
            "request_failed exception method=%s path=%s qs=%s duration_ms=%s",
            request.method,
            path,
            request.url.query,
            ms,
        )
        raise

    if response.status_code >= 400:
        ms = int((time.perf_counter() - t0) * 1000)
        _http_fail_logger.warning(
            "request_failed status=%s method=%s path=%s qs=%s duration_ms=%s",
            response.status_code,
            request.method,
            path,
            request.url.query,
            ms,
        )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(files.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(training.models_router)
app.include_router(predictions.router)
app.include_router(agent.router)
app.include_router(kb.router)
app.include_router(simulate.router)
app.include_router(runs.router)

meta.register_meta_routes(app)
