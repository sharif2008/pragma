# Plan: backend storage vs `experiments/<task>/`

## Live layout

Web-app files stay in `backend/storage/`. Console stages write under `experiments/<task>/`. **Keep only the last result** for each case (`run_*` or purpose-named live folder); older complete trees go to `archive/`.

| Case | Live path |
|------|-----------|
| Train | `experiments/detect-train/run_*/` |
| Predict | `experiments/detect-predict/run_*/` |
| RAG index | `experiments/rag-index/vector_store/` |
| Reason ablation | `experiments/reason/reason_ablation_9/` |
| Gold | `experiments/gold-100/` |
| RAG eval100 | `experiments/rag/rag_eval100/` |
| Reason 1000 | `experiments/reason/rag_reason_1000/` |
| Fixtures | `experiments/fixtures/<purpose>/` |

## Helpers (`env.py`)

- `experiment_dir(task)`, `new_run_dir` / `archive_older_runs` for train/predict
- `new_named_live_dir` / `LIVE_NAMED_DIRS` for reason / gold-100 / rag
- `resolve_model_dir()`, `resolve_latest_predict_dir()`, `resolve_fixture_csv()`

## Do not

- Mix API joblib and paper `.pth` in `backend/storage/models/`
- Put flow CSVs under `detect-predict/`
- Write console dumps into `backend/storage/`

## Done when

Each pipeline stage has **one** live result folder; cwd does not change the root; UI still only uses `backend/storage/`.
