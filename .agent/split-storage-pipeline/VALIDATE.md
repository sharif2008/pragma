# Validate: experiments-per-task plan

Checked against the tree on 2026-10-08. Updated layout (`experiments/<task>/`) still passes.

## Confirmed problem

| Location | What is mixed |
|----------|----------------|
| `backend/storage/models/` | API `model_*.joblib` **and** paper `vfl_model_best.pth` |
| Repo `outputs/` | `detect_predict` CSV/JSON (cwd-sensitive) |
| `RAG_docs/` (relative) | `rag_build.py` / `reason.py` assume repo-root cwd |
| `backend/run/output/` | Console e2e ledgers next to API code |

`env.py` points `MODEL_DIR` at backend storage and `PAPER_OUTPUT_DIR` at repo `outputs/`. That is what this action fixes.

## Why one folder per task

`pipeline.py` stages already are `detect-train`, `detect-predict`, `rag-index`, `reason`, `evaluate`, `e2e`. Mapping `experiment_dir(stage)` 1:1 avoids a second naming scheme.

Write isolation is valid: predict must not overwrite train checkpoints. Cross-task **reads** are required (predict ← train weights; reason ← rag-index + detect-predict). Spec states that explicitly.

## Launch-context evidence

- FastAPI `Settings.storage_root` — leave as backend.
- `pipeline.py` `chdir` to repo root is a hidden dependency; absolute `experiments/<task>/` removes it.
- `detect_predict.py` already uses path helpers; retarget to `experiment_dir("detect-train")` / `experiment_dir("detect-predict")`.
- `rag_build.py` / `reason.py` still use `Path("RAG_docs/...")` — switch to `experiment_dir("rag-index")` and `experiment_dir("reason")`.
- `e2e` HTTP → backend storage; console ledger → `experiments/e2e/`.

## Shared catalogs / HF cache

Unchanged: catalogs stay in `backend/storage/*.json`. `hf_home` stays backend. Do not put a second cache under `experiments/`.

## Joblib vs .pth

API `.joblib` → `backend/storage/models/`. Paper `.pth` → `experiments/detect-train/`. `resolve_model_dir()` looks at the train task folder first, then old backend path for one-release compat.

## Git

Ignore `experiments/**`; keep per-task `.gitkeep`. Track `.agent/`. Do **not** ignore `experiments/` as a single blob without gitkeeps or the empty tree disappears.

## Risks

| Risk | Mitigation |
|------|------------|
| Predict cannot find `.pth` after retarget | Fallback to `backend/storage/models/*.pth`, then migrate |
| Reason still reads `RAG_docs/predictions` | Read `experiments/detect-predict/`; write `experiments/reason/` |
| `STORAGE_ROOT` pointed at experiments | Document: `STORAGE_ROOT` = API; `EXPERIMENTS_ROOT` = CLI |
| Task name drift (`detect_train` vs `detect-train`) | Use exact `pipeline.py` stage keys (hyphens) |

## Verdict

**Pass.** `experiments/<task>/` is a better fit than a flat `storage_pipeline/` because it matches console stages (train, detect, reason) and keeps each task’s outputs from overwriting the others.
