# Plan: train current features → store → predict

## Live result

Keep the **latest** train and predict runs. A new `pipeline.py detect-train` writes `experiments/detect-train/run_<ts>/` and moves the previous live `run_*` to `archive/`. Same pattern for detect-predict.

Current:

- Train: `experiments/detect-train/run_20261008_030606/`
- Predict: `experiments/detect-predict/run_20261008_030634/`

## Re-run

```powershell
cd D:\Projects\ChainAgentVFL\backend
.\.venv\Scripts\Activate.ps1
python scripts/pipeline.py detect-train
python scripts/pipeline.py detect-predict
```

Predict **must not** train. It reloads `resolve_model_dir()`. Do not copy `.pth` into `backend/storage/models/`.

## Sanity

- `model_metadata.json` feature lists match `agentic_features.json` `logged_features` (23 / 32 / 42)
- `agent_names` are Access / ISP, Perimeter / IDS, Endpoint / EDR
- `model_comparison_*.json` has both confusion matrices
- Predict CSV/JSON live only under the latest `detect-predict/run_*/`

## Done when

One live train folder and one live predict folder; comparison file includes both CMs.
