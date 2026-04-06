-- Run once on existing MySQL databases (SQLAlchemy create_all does not add columns to existing tables).
-- Example: mysql -u USER -p DBNAME < scripts/alter_prediction_jobs_add_results_json.sql

ALTER TABLE prediction_jobs
  ADD COLUMN results_json JSON NULL
  COMMENT 'Per-row predictions, class probabilities, SHAP payloads'
  AFTER config_json;
