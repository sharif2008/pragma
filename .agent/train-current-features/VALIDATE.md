# Validate: train-current-features

Checked 2026-10-08 against code + `experiments/detect-train/model_metadata.json`.

## Retrain is required

Copied metadata `agent1_features` starts with `udps.srcdst_rst_packet_count`, … (heuristic-era lists). Current JSON Access/ISP starts with `bidirectional_duration_ms`, `src2dst_duration_ms`, … **Mismatch confirmed.** Predict on the old `.pth` cannot be relabeled into the current partition.

`detect_train.py` already calls `load_agent_definitions(AGENTIC_FEATURES_JSON)` then `split_features_by_agent_definitions`. A fresh `pipeline.py detect-train` **will** train current columns. No new trainer is needed.

## Model folder reuse already wired

`resolve_model_dir()` prefers `experiments/detect-train/` when `vfl_model_best.pth` is there. After retrain, detect-predict will pick it up without a second path change.

## Comparison file gap

`model_comparison_*.json` already stores Acc / macro-recall / macro-F1 for VFL vs Standard_NN. Confusion matrices are **printed** (`cm`, `cm_standard`) but **not** stored. Spec’s extra fields are a small edit in `detect_train.py` before the run, not a new script.

## Display names

`load_agent_definitions` still sets `agent_names` to `["RAN", "Edge", "Core"]`. `pragma_domain` exists on each agent. One-line mapping is enough for metadata and SHAP keys to match the paper domains. JSON **keys** in `agentic_features.json` stay RAN/Edge/Core (API compat). Pass.

## Runtime constraints

- Dataset `datasets/undersampled_CIC2017_dataset.csv` is present (~140 MB).
- Train is CPU/GPU heavy (up to 100 epochs + KernelSHAP). Plan treats that as a **run** step, not a code gap.
- Centralized NN is still evaluation-only (not saved). Matches paper: baseline is not the live detector.

## Verdict

**Pass.** Code path for current features already exists; the old checkpoint is stale. Remaining work: domain names in metadata, persist confusion matrices in the comparison file, then run train → predict.
