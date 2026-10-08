# Results: train-current-features

Validated 2026-10-08 after `pipeline.py detect-train` then `detect-predict`.

## Train (`experiments/detect-train/`)

- `model_metadata.json` `agent_names`: Access / ISP, Perimeter / IDS, Endpoint / EDR
- Feature counts **23 / 32 / 42**, lists **match** `agentic_features.json` `logged_features`
- Checkpoint: `vfl_model_best.pth` (best epoch 99, val F1 0.8598)

Held-out test (N=78,080), from `model_comparison_20261008_025340.json`:

| Model | Acc. | Macro Rec. | Macro F1 |
|-------|------|------------|----------|
| VFL | 0.9597 | 0.9855 | 0.8635 |
| Standard NN | 0.9572 | 0.9822 | 0.8409 |
| Difference (VFL − NN) | +0.0024 | +0.0033 | +0.0226 |

Confusion matrices and per-class reports are in that JSON plus `confusion_vfl_*.csv` / `confusion_nn_*.csv`.

## Predict reuse

`detect-predict` loaded **`experiments/detect-train`** (not `backend/storage/models`). Wrote `experiments/detect-predict/predictions_20261008_030634.csv`.

## Verdict

**Pass** against SPEC.md done-when items: current partition trained, model folder reused, comparison file includes both CMs.
