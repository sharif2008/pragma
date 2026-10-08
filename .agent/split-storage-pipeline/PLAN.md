# Plan: backend storage vs `experiments/<task>/`

## Goal

Web-app files stay in `backend/storage/`. Console train / detect / reason / evaluate write under **`experiments/<pipeline-stage>/`**, one folder per task, independent of cwd.

## Approach

1. **`env.py` owns roots**
   - `BACKEND_STORAGE` = `backend/storage` (`STORAGE_ROOT`)
   - `EXPERIMENTS_ROOT` = repo `experiments/` (env `EXPERIMENTS_ROOT`)
   - `experiment_dir(task: str) -> Path` mkdir-on-use
   - Task names = `pipeline.py` stages: `detect-train`, `detect-predict`, `rag-index`, `reason`, `evaluate`, `e2e`
   - `resolve_model_dir()` for CLI: `experiments/detect-train/` first, then old `backend/storage/models` if a `.pth` is still there

2. **`pipeline.py`**
   ```python
   os.environ["CHAINAGENT_SCOPE"] = "pipeline"
   os.environ.setdefault("CHAINAGENT_TASK", stage)
   ```
   Absolute experiment paths so running from `backend/scripts/` still lands in `experiments/<task>/`.

3. **Retarget scripts** (destination only):

   | Script | Writes to | Reads from |
   |--------|-----------|------------|
   | `detect_train.py` | `experiments/detect-train/run_<ts>/` (old runs → `archive/`) | `datasets/`, catalogs |
   | `detect_predict.py` | `experiments/detect-predict/run_<ts>/` (old runs → `archive/`) | latest train `run_*`, `inputs/` |
   | `rag_build.py` | `experiments/rag-index/` | CLI knowledge dir (default under that task folder) |
   | `reason.py` | `experiments/reason/` | `experiments/rag-index/`, `experiments/detect-predict/` |
   | `evaluate.py` | `experiments/evaluate/` | train/predict/reason as needed |
   | `attack_monitor.py` ledger | `experiments/e2e/` | API (`backend/storage` via HTTP) |

4. **FastAPI unchanged** except paper `.pth` must not be the API `models/` contract. Joblib stays `backend/storage/models/model_<uuid>.joblib`.

5. **Migrate existing console artifacts**
   - `.pth` / scalers / metadata / shap_background → `experiments/detect-train/`
   - repo `outputs/predictions_*` → `experiments/detect-predict/`
   - `RAG_docs/` if present → `experiments/rag-index/`
   - `backend/run/output/` → `experiments/e2e/` (optional)

6. **Gitignore** `experiments/**` with per-task `.gitkeep`. Keep `.agent/` tracked.

7. **README** CLI section: artifacts live in `experiments/<task>/`.

## Non-goals

- RAN/Edge/Core rename
- Retrain / architecture change
- Second HuggingFace cache
- Moving `datasets/` or `backend/run/data/`

## Implementation order

1. `env.py`: `EXPERIMENTS_ROOT`, `experiment_dir(task)`
2. `pipeline.py`: set scope + task
3. detect-train / detect-predict
4. rag-index / reason / evaluate
5. e2e ledger path
6. gitignore + gitkeeps
7. README + migrate note

## Done when

- `python scripts/pipeline.py detect-train` writes only under `experiments/detect-train/`
- `python scripts/pipeline.py detect-predict` writes under `experiments/detect-predict/` and loads weights from `experiments/detect-train/`
- `python scripts/pipeline.py reason` writes under `experiments/reason/`
- UI still only uses `backend/storage/`
- Cwd does not change the root
