# Spec: backend storage vs experiments (per task)

Action: `split-storage-pipeline`

Two on-disk roots. Scripts pick a root from launch context, not from cwd.

## Roots

| Root | Path | Owner |
|------|------|--------|
| Backend | `backend/storage/` (`STORAGE_ROOT`) | FastAPI / UI |
| Experiments | `experiments/` (`EXPERIMENTS_ROOT`) | Console training / detect / reason / evaluate |

Repo root `.agent/` is **plans/specs only**, not runtime artifacts.

## Backend (`backend/storage/`) — web app

Keep using `Settings.storage_root`. Console dumps do not go here.

- `uploads/`, `knowledge/`, `training_datasets/`
- `models/model_<job-uuid>.joblib` (API train only)
- `predictions/`, `reports/`, `vector_db/`
- `hf_home/` (shared download cache; both sides may *read*)
- Catalogs (read-only for CLI): `agentic_features.json`, `attack_options.json`, `base_docs/`

## Experiments (`experiments/<task>/`) — console

One **task folder** per `pipeline.py` stage. A script **writes only inside its own folder** and may only overwrite files in that folder. It may **read** another task folder (train → predict, index → reason). It must not archive or replace a different task’s directory.

**One live result per case.** Timestamped `run_*` (train/predict) or purpose-named live folders (reason/rag). A new run of the *same* task may overwrite that task’s current files in place (gold-100 at the task root) or archive a prior `run_*` under that task’s `archive/`. Shared inputs live in `experiments/fixtures/<owner-task>/` (other tasks may read). Results live in `experiments/<task>/` and must not overwrite another task’s folder.

```
experiments/
  detect-train/run_<ts>/          # latest VFL + NN comparison; older → archive/
  detect-predict/run_<ts>/        # latest inference + SHAP; older → archive/
  rag-index/vector_store/         # current FAISS (overwrite in place)
  reason/reason_ablation_9/       # latest 9×3 plans; older → archive/
  gold-100/                       # gold detect + freeze (task root)
  rag/rag_eval100/                # gold-core RAG eval (when run)
  reason/rag_reason_1000/         # 1000-row reason set (when run)
  evaluate/
  e2e/
  fixtures/<owner-task>/          # shared inputs (read by other tasks)
```

| Task | Script | Live write |
|------|--------|------------|
| `detect-train` | `detect_train.py` | `run_<ts>/` (checkpoint + reports); prior live → `archive/` |
| `detect-predict` | `detect_predict.py` | `run_<ts>/`; prior → `archive/` |
| `rag-index` | `rag_build.py` | `knowledge/`, `vector_store/` |
| `reason` | `reason_ablation.py` | `reason_ablation_9/` (`reason.py` still uses `action_plans/`) |
| `gold-100` | gold freeze | task root (`experiments/gold-100/`) |
| `rag` | rag eval | `rag_eval/` |
| `evaluate` | `evaluate.py` | metric tables / plots |
| `e2e` | `attack_monitor.py` | run ledgers |

Cross-task reads (not copies):

- `detect-predict` reads the **latest** `experiments/detect-train/run_*/` checkpoint (`resolve_model_dir()`)
- `reason` reads `experiments/rag-index/vector_store/` and the **latest** `experiments/detect-predict/run_*/` (`resolve_latest_predict_dir()`)
- `evaluate` reads train/predict/reason outputs as needed

Helper: `experiment_dir("detect-train")` → `REPO/experiments/detect-train`. Purpose lives: `LIVE_NAMED_DIRS` in `env.py`.

## Shared inputs (not generated)

- `datasets/` — training CSVs
- `experiments/fixtures/<owner-task>/` — `gold-100`, `reason_ablation_9`, `rag_reason_1000`, `rag_retrieval_test_50` (pointers: `blockchain_all_1000`, `e2e_detect_to_blockchain_1000`)
- `backend/run/data/sample.csv` — API fixture fallback
- catalogs under `backend/storage/*.json`

## Context rule

```
CHAINAGENT_SCOPE=pipeline    # set by pipeline.py before runpy
CHAINAGENT_SCOPE=backend     # FastAPI / uvicorn (default)
EXPERIMENTS_ROOT=<path>      # optional override of repo/experiments
```

If unset: FastAPI always uses `STORAGE_ROOT`. Paper scripts default writes to `experiments/<task>/`. `e2e` HTTP still persists API artifacts in backend storage; the **console ledger** goes to `experiments/e2e/`.
