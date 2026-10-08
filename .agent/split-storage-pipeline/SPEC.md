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

One **task folder** per `pipeline.py` stage. A script **writes only inside its own folder**. It may **read** another task folder (train → predict, index → reason).

```
experiments/
  detect-train/       # VFL + centralized NN training
  detect-predict/     # inference + SHAP details
  rag-index/          # CLI knowledge corpus + FAISS
  reason/             # mitigation plans
  evaluate/           # scoring tables vs centralized NN
  e2e/                # attack_monitor / network_monitor console ledgers
```

| Task | Script | Writes |
|------|--------|--------|
| `detect-train` | `detect_train.py` | `run_<ts>/` (checkpoint + reports); prior live runs moved to `archive/` |
| `detect-predict` | `detect_predict.py` | `run_<ts>/` (prediction CSV/JSON); `inputs/` stays; prior runs → `archive/` |
| `rag-index` | `rag_build.py` | `knowledge/`, `vector_store/` |
| `reason` | `reason.py` | `action_plans/` |
| `evaluate` | `evaluate.py` | metric tables / plots |
| `e2e` | `attack_monitor.py` | run ledgers (`report.json`, `ledger.json`, …) |

Cross-task reads (not copies):

- `detect-predict` reads the **latest** `experiments/detect-train/run_*/` checkpoint (`resolve_model_dir()`)
- `reason` reads `experiments/rag-index/vector_store/` and the **latest** `experiments/detect-predict/run_*/` (`resolve_latest_predict_dir()`)
- `evaluate` reads train/predict/reason outputs as needed

Helper: `experiment_dir("detect-train")` → `REPO/experiments/detect-train`.

## Shared inputs (not generated)

- `datasets/` — training CSVs
- `backend/run/data/sample.csv` — fixture CSV
- catalogs under `backend/storage/*.json`

## Context rule

```
CHAINAGENT_SCOPE=pipeline    # set by pipeline.py before runpy
CHAINAGENT_SCOPE=backend     # FastAPI / uvicorn (default)
EXPERIMENTS_ROOT=<path>      # optional override of repo/experiments
```

If unset: FastAPI always uses `STORAGE_ROOT`. Paper scripts default writes to `experiments/<task>/`. `e2e` HTTP still persists API artifacts in backend storage; the **console ledger** goes to `experiments/e2e/`.
