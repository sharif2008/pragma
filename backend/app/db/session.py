"""Database engine and session factory."""

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.base import Base

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_prediction_jobs_results_json() -> None:
    """Add ``results_json`` when missing (``create_all`` does not ALTER existing tables)."""
    if engine.dialect.name != "mysql":
        return
    with engine.begin() as conn:
        cnt = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = 'prediction_jobs' AND COLUMN_NAME = 'results_json'"
            )
        ).scalar_one()
        if int(cnt) == 0:
            conn.execute(
                text(
                    "ALTER TABLE prediction_jobs "
                    "ADD COLUMN results_json JSON NULL "
                    "COMMENT 'Per-row predictions, class probabilities, SHAP payloads' "
                    "AFTER config_json"
                )
            )


def _ensure_agentic_reports_results_row_index() -> None:
    """Add ``results_row_index`` when missing (existing MySQL DBs pre-migration)."""
    if engine.dialect.name != "mysql":
        return
    with engine.begin() as conn:
        cnt = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = 'agentic_reports' AND COLUMN_NAME = 'results_row_index'"
            )
        ).scalar_one()
        if int(cnt) == 0:
            conn.execute(
                text(
                    "ALTER TABLE agentic_reports "
                    "ADD COLUMN results_row_index INT NULL "
                    "COMMENT 'results_row_index from POST /agent/decide' "
                    "AFTER prediction_job_id"
                )
            )


def _ensure_agentic_reports_agentic_job_id() -> None:
    """Add ``agentic_job_id`` when missing (FK to agentic_jobs)."""
    if engine.dialect.name != "mysql":
        return
    with engine.begin() as conn:
        cnt = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = 'agentic_reports' AND COLUMN_NAME = 'agentic_job_id'"
            )
        ).scalar_one()
        if int(cnt) == 0:
            conn.execute(
                text(
                    "ALTER TABLE agentic_reports "
                    "ADD COLUMN agentic_job_id INT NULL "
                    "COMMENT 'FK to agentic_jobs.id when run used a registered handoff' "
                    "AFTER results_row_index"
                )
            )


def init_db() -> None:
    """Create tables (development convenience; use Alembic in production)."""
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_prediction_jobs_results_json()
    _ensure_agentic_reports_results_row_index()
    _ensure_agentic_reports_agentic_job_id()
    _ensure_agentic_report_trust_anchors_table()


def _ensure_agentic_report_trust_anchors_table() -> None:
    """Create trust anchor table for existing MySQL DBs if missing."""
    if engine.dialect.name != "mysql":
        return
    with engine.begin() as conn:
        cnt = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'agentic_report_trust_anchors'"
            )
        ).scalar_one()
        if int(cnt) == 0:
            # Multi-worker startup can race; IF NOT EXISTS makes this idempotent.
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS agentic_report_trust_anchors ("
                    "id INT AUTO_INCREMENT PRIMARY KEY,"
                    "agentic_report_id INT NOT NULL UNIQUE,"
                    "chain_id INT NOT NULL DEFAULT 31337,"
                    "contract_address VARCHAR(128) NOT NULL DEFAULT '',"
                    "tx_hash VARCHAR(128) NOT NULL DEFAULT '',"
                    "payload_version VARCHAR(32) NOT NULL DEFAULT 'v1',"
                    "commitment_sha256 VARCHAR(64) NOT NULL DEFAULT '',"
                    "agent_key_sha256 VARCHAR(64) NOT NULL DEFAULT '',"
                    "report_key_sha256 VARCHAR(64) NOT NULL DEFAULT '',"
                    "anchored_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),"
                    "error TEXT NULL,"
                    "INDEX(agentic_report_id),"
                    "CONSTRAINT fk_anchor_report FOREIGN KEY (agentic_report_id) "
                    "REFERENCES agentic_reports(id) ON DELETE CASCADE"
                    ")"
                )
            )
