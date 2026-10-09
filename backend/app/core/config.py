"""Application settings loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from scripts.rag_chunking import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_TOP_K


_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ChainAgentVFL API"
    debug: bool = False

    # MySQL: mysql+pymysql://user:password@host:3306/dbname
    # Create DB: mysql -u root -p < db/init_mysql.sql (database name uses a hyphen)
    database_url: str = Field(
        default="mysql+pymysql://root:test@127.0.0.1:3306/agentic-vfl",
        alias="DATABASE_URL",
    )

    # Single root for all offline artifacts (data, models, predictions, reports, vectors)
    storage_root: Path = Field(
        default=Path(__file__).resolve().parent.parent.parent / "storage",
        alias="STORAGE_ROOT",
    )

    # Application log directory (default: backend/logs/)
    log_dir: Path = Field(default=_BACKEND_DIR / "logs", alias="LOG_DIR")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # HuggingFace cache root (models, tokenizers, etc.). Keeping this stable avoids re-downloads.
    hf_home: Path = Field(default=_BACKEND_DIR / "storage" / "hf_home", alias="HF_HOME")

    rag_chunk_size: int = RAG_CHUNK_SIZE
    rag_chunk_overlap: int = RAG_CHUNK_OVERLAP
    rag_top_k: int = RAG_TOP_K

    # Training / prediction
    default_test_size: float = 0.2
    default_random_state: int = 42

    # Trust anchoring (local Hardhat / JSON-RPC)
    trust_chain_enabled: bool = Field(default=False, alias="TRUST_CHAIN_ENABLED")
    trust_chain_rpc_url: str = Field(default="http://127.0.0.1:8545", alias="TRUST_CHAIN_RPC_URL")
    trust_chain_private_key: str | None = Field(default=None, alias="TRUST_CHAIN_PRIVATE_KEY")
    trust_chain_contract_address: str | None = Field(default=None, alias="TRUST_CHAIN_CONTRACT_ADDRESS")
    trust_chain_chain_id: int = Field(default=31337, alias="TRUST_CHAIN_CHAIN_ID")
    trust_chain_payload_version: str = Field(default="v1", alias="TRUST_CHAIN_PAYLOAD_VERSION")

    # Per-tier execution agents (HTTP POST after on-chain markApplied). Empty = stub receipt only.
    exec_agent_access_url: str | None = Field(default=None, alias="EXEC_AGENT_ACCESS_URL")
    exec_agent_perimeter_url: str | None = Field(default=None, alias="EXEC_AGENT_PERIMETER_URL")
    exec_agent_endpoint_url: str | None = Field(default=None, alias="EXEC_AGENT_ENDPOINT_URL")
    exec_agent_timeout_s: float = Field(default=5.0, alias="EXEC_AGENT_TIMEOUT_S")
    # Human reviewer key (signs revisePlan for human corrections).
    trust_chain_reviewer_private_key: str | None = Field(default=None, alias="TRUST_CHAIN_REVIEWER_PRIVATE_KEY")
    # One executor key per network tier (each signs markApplied for its own tier only).
    executor_access_private_key: str | None = Field(default=None, alias="EXECUTOR_ACCESS_PRIVATE_KEY")
    executor_perimeter_private_key: str | None = Field(default=None, alias="EXECUTOR_PERIMETER_PRIVATE_KEY")
    executor_endpoint_private_key: str | None = Field(default=None, alias="EXECUTOR_ENDPOINT_PRIVATE_KEY")
    # When set (e.g. http://127.0.0.1:8000), the orchestrator calls tier executors over HTTP;
    # otherwise it invokes the same executor handlers in-process.
    executor_base_url: str | None = Field(default=None, alias="EXECUTOR_BASE_URL")
    # Agent self-correction retries when a plan fails validation before storePlan.
    agent_plan_max_retries: int = Field(default=2, alias="AGENT_PLAN_MAX_RETRIES")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_storage_dirs(settings: Settings) -> None:
    """Create storage subdirectories if missing."""
    root = settings.storage_root
    subdirs = (
        root / "uploads",
        root / "knowledge",
        root / "models",
        root / "predictions",
        root / "reports",
        root / "vector_db",
        root / "training_datasets",
    )
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)

    settings.log_dir.mkdir(parents=True, exist_ok=True)

    # Ensure a stable local HuggingFace cache directory.
    settings.hf_home.mkdir(parents=True, exist_ok=True)
    # Only set if not already defined by the environment (let ops override).
    os.environ.setdefault("HF_HOME", str(settings.hf_home))
