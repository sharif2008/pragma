# Spec: train on current features, reuse for predict, write comparison matrix

Action: `train-current-features`

Retrain the paper VFL (and centralized NN baseline) on the **current** `agentic_features.json` partition. Persist the checkpoint under `experiments/detect-train/`. Predict reloads that folder. Comparison metrics **and confusion matrices** land in one report file in the same train folder.

## Current features (source of truth)

`backend/storage/agentic_features.json` schema 1.2:

| JSON key (compat) | Domain label | Columns |
|-------------------|--------------|---------|
| RAN | Access / ISP | 23 volume/rate |
| Edge | Perimeter / IDS | 32 ports / src2dst size-timing |
| Core | Endpoint / EDR | 42 bidirectional + reverse-path |

97 listings / 88 unique columns (9 shared on purpose). This is **not** the split baked into the old `model_metadata.json`.

Display names in new metadata: **Access / ISP**, **Perimeter / IDS**, **Endpoint / EDR** (`pragma_domain`), not RAN / Edge / Core.

## Model folder (write + reuse)

**Keep only the latest live run** `experiments/detect-train/run_*/` (via `resolve_model_dir()`). A new train writes a new `run_<ts>/` and moves the previous live run into `archive/`. Current: `run_20261008_030606`.

| File | Role |
|------|------|
| `vfl_model_best.pth` | VFL weights + `model_config` |
| `meta_model_best.pth` | distilled SHAP meta-model |
| `scaler1.pkl` `scaler2.pkl` `scaler3.pkl` | per-party scalers |
| `shap_background.npy` | KernelSHAP background |
| `model_metadata.json` | feature lists, label map, **pragma_domain** names, train metrics |
| `model_comparison_<ts>.json` | VFL vs centralized NN + **confusion matrices** |
| `model_comparison_<ts>.csv` | metric table |
| `model_comparison_report_<ts>.txt` | human-readable report |

Predict **must not** train. It only reads this folder (`python scripts/pipeline.py detect-predict` → latest `experiments/detect-predict/run_*/`; current `run_20261008_030634`).

## Comparison matrix file

`experiments/detect-train/run_*/model_comparison_<timestamp>.json` must include, for **VFL** and **Standard_NN** on the same held-out test split:

- accuracy, macro-recall, macro-F1
- per-class precision / recall / F1
- **confusion matrix** (rows = true, cols = predicted, with `class_names`)
- difference row (VFL − NN)

CSV/txt are projections of the same numbers. Predict-time label vs truth (if the sample CSV has `label`) stays under `experiments/detect-predict/`; it does not replace the train comparison file.

## Inputs

- Dataset: `datasets/*.csv` (present: `undersampled_CIC2017_dataset.csv`)
- Catalog: `backend/storage/agentic_features.json`
- Sample predict CSV: `experiments/fixtures/reason_ablation_9/` (`resolve_sample_csv()`), then `backend/run/data/sample.csv`

## Out of scope

- API `.joblib` under `backend/storage/models/`
- Renaming JSON **keys** RAN/Edge/Core in `agentic_features.json` (compat); only stored **display** names change
- Saving the centralized NN `.pth` (comparison-only; not on the live detect path)
