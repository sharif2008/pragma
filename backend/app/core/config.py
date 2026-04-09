"""Application settings loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k: int = 5

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
