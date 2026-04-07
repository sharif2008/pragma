"""FastAPI entrypoint: modular ML + RAG + agentic API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
