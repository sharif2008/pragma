# Plan: train current features → store → predict → comparison matrix

## Why

The checkpoint now in `experiments/detect-train/` was copied from an older run. Its `agent1/2/3_features` do **not** match current `logged_features`. Detect still works, but it is not “current features.” Retrain from `agentic_features.json`, keep artifacts in the train model folder, reload them on predict, and persist VFL vs NN **confusion matrices** in the comparison file (today they print to stdout only).

## Steps

1. **Names in metadata** — `load_agent_definitions()` (or detect_train after load) set `agent_names` from each agent’s `pragma_domain` (Access / ISP, Perimeter / IDS, Endpoint / EDR). Keep JSON key order RAN → Edge → Core so the 23/32/42 split is unchanged.

2. **Confusion matrices in the comparison artifact** — after both test evaluations, add to `vfl_results` / `standard_results` (and thus `model_comparison_*.json`):
   - `confusion_matrix`: nested list of ints
   - `class_names`: label order
   - `classification_report`: sklearn `output_dict=True`
   Mirror a compact table in the `.txt` report and optional extra CSV `confusion_vfl_<ts>.csv` / `confusion_nn_<ts>.csv` (same folder).

3. **Train** (long run; no uvicorn):

   ```powershell
   cd D:\Projects\ChainAgentVFL\backend
   .\.venv\Scripts\Activate.ps1
   python scripts/pipeline.py detect-train
   ```

   Writes only under `experiments/detect-train/`. Overwrites `vfl_model_best.pth` (intended).

4. **Predict reuses that folder** — no path changes if `resolve_model_dir()` already prefers `experiments/detect-train/` when `vfl_model_best.pth` exists.

   ```powershell
   python scripts/pipeline.py detect-predict
   ```

   Writes under `experiments/detect-predict/`.

5. **Sanity check the report file** — open `experiments/detect-train/model_comparison_*.json` and confirm:
   - feature counts 23 / 32 / 42 (or 97 listings)
   - `agent_names` are domain labels
   - both confusion matrices present
   - Acc / macro-F1 for VFL and Standard_NN

## Implementation order (code, then run)

1. `scripts/vfl.py` `load_agent_definitions`: expose `pragma_domain` as `agent_names` (fallback to key).
2. `detect_train.py`: attach CM + per-class report to `vfl_results` / `standard_results`; print domain names.
3. Run `detect-train` (hours possible: 100 epochs + SHAP).
4. Run `detect-predict` to prove reuse.
5. Do not copy new `.pth` into `backend/storage/models/` (API stays joblib-only).

## Done when

- `experiments/detect-train/model_metadata.json` feature lists match `agentic_features.json` `logged_features`.
- Same folder holds `vfl_model_best.pth` used by detect-predict.
- `model_comparison_*.json` contains both confusion matrices and the VFL vs NN metric table.
- `experiments/detect-predict/` has a new timestamped CSV/JSON from that checkpoint.
