-- Run from backend/: mysql -u root -p < db/init_mysql.sql
-- Tables are created by SQLAlchemy ORM (app.models.domain) on app startup or: python scripts/init_orm_tables.py

CREATE DATABASE IF NOT EXISTS `agentic-vfl`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- Note: tables are created by SQLAlchemy on app startup.
-- The trust anchoring table `agentic_report_trust_anchors` is created/ensured by app/db/session.py for existing DBs.
