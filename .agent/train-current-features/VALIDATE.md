# Validate: train-current-features

Checked 2026-10-08 after `pipeline.py detect-train` then `detect-predict`. Keep-last: one live run each.

## Train

`experiments/detect-train/run_20261008_030606/`

- `agent_names`: Access / ISP, Perimeter / IDS, Endpoint / EDR
- Feature counts **23 / 32 / 42**, lists match `agentic_features.json` `logged_features`
- `vfl_model_best.pth` (best epoch 99, val F1 0.8598)
- Comparison + confusion matrices in `model_comparison_20261008_025340.json`

## Predict

`experiments/detect-predict/run_20261008_030634/`

Loaded the live train folder (not `backend/storage/models`). Wrote `predictions_20261008_030634.csv` + detailed JSON.

## Verdict

**Pass.** Current partition trained; model folder reused; one live result per stage.
