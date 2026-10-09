# Validate: experiments-per-task plan

Checked 2026-10-08 against the live tree after keep-last cleanup.

## Live results (one each)

| Task | Live |
|------|------|
| detect-train | `experiments/detect-train/run_20261008_030606/` |
| detect-predict | `experiments/detect-predict/run_20261008_030634/` |
| rag-index | `experiments/rag-index/vector_store/` |
| reason | `experiments/reason/reason_ablation_9/` |
| gold-100 / rag | gold results in `experiments/gold-100/`; rag eval not run |
| fixtures | `experiments/fixtures/<purpose>/` |

Older reason 9×6: `experiments/reason/archive/`.

## Isolation

- Predict does not overwrite train checkpoints.
- Reason reads train/predict/index; writes only under `experiments/reason/`.
- API joblib stays `backend/storage/models/`. Paper `.pth` stays under detect-train.
- Catalogs + `hf_home` stay in backend storage.

## Verdict

**Pass.** One live result per executed case; gold-100 writes at the task root; fixtures are named by owner task.
