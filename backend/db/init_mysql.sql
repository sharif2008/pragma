-- Run from backend/: mysql -u root -p < db/init_mysql.sql
-- Tables are created by SQLAlchemy ORM (app.models.domain) on app startup or: python scripts/init_orm_tables.py

CREATE DATABASE IF NOT EXISTS `agentic-vfl`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
