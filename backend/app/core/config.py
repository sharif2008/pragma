"""Application settings loaded from environment variables."""

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

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k: int = 5

    # Training / prediction
    default_test_size: float = 0.2
    default_random_state: int = 42


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
