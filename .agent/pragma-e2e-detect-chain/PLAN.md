# Plan: 1000-flow happy-path e2e

Action: `pragma-e2e-detect-chain`. Source of truth: [`SPEC.md`](SPEC.md). This plan only sequences that spec.

## Why

Measure success-path latency on 1000 flows: detect → ranked RAG → one Mitigation Plan → unmodified `storePlan` / verify / `markApplied`.

No inject. No BERTScore. No other-task files. Does **not** use `experiments/reason/`.

## Independent

This job opens the frozen `--input` CSV (default `experiments/data/e2e-detect-chain/`) plus the live system (VFL checkpoint, `experiments/rag-index/vector_store/`, OpenAI, Hardhat).

Do not read gold JSON, eval100 jsonl, agentic-attack receipts, or any other fixture. Do not import `agentic_attack_eval.py`. Write only to `experiments/e2e-detect-chain/` (except `--sample-only` into this fixture).

## Prerequisite (SPEC)

- `experiments/data/e2e-detect-chain/flows.csv` (create with `--sample-only` if missing)
- Current VFL checkpoint and `experiments/rag-index/vector_store/`
- `OPENAI_API_KEY` + `OPENAI_MODEL=gpt-6-luna`
- Hardhat already on `:8545`; `TRUST_CHAIN_*` in `backend/.env`

## Steps

1. **Fixture** — water-fill 1000 from the seed-42 test split into this fixture. Stop if N ≠ 1000.
2. **Detect** — chunked `detect_predict.py` into `experiments/e2e-detect-chain/detect/`.
3. **Reason** — `RAG_RANKING` only. `hybrid_children` (FAISS 80 + BM25 80 → RRF → MMR λ=0.5 → 20 children → 5 parents). One LLM call per flow. Resume `runs.jsonl`. Write `predicted_label` on every mitigation-plans case.
4. **Chain** — store the generated plan unchanged. `planId={split_index}-RAG_RANKING-honest`. `storePlan` → verify → `markApplied`. Record ms. unresolved = 0. Continue on other reverts. No mutation, no inject.
5. **Report** — write the SPEC latency matrix (every step × 9 attack types, **Overall last**) and the honest-store matrix into `experiments/e2e-detect-chain/report.md` and `.agent/pragma-e2e-detect-chain/REPORT.md`. Cells are total ms. E2E is the row sum of Detect…Apply. Overall is the column sum of the nine attack types. `--report-only` rebuilds both from this folder’s jsonl.

```
cd backend
python scripts/e2e_detect_chain.py
python scripts/e2e_detect_chain.py --input ../experiments/data/eval-10
python scripts/e2e_detect_chain.py --sample-only
python scripts/e2e_detect_chain.py --predict-only
python scripts/e2e_detect_chain.py --reason-only
python scripts/e2e_detect_chain.py --chain-only
python scripts/e2e_detect_chain.py --report-only
```

Leave Hardhat running for `--chain-only` / default.

## Done when

```
experiments/data/e2e-detect-chain/
  flows.csv              # N=1000
  manifest.json
  README.md

experiments/e2e-detect-chain/
  detect/predictions_detailed.json
  runs.jsonl             # 1000
  mitigation_plans.json
  honest.jsonl
  latency.json           # overall + 9 labels, each step sum/mean/min/max + n
  report.md              # latency matrix + E2E mean/min/max
  manifest.json
  README.md

.agent/pragma-e2e-detect-chain/REPORT.md   # same latency matrix
```

1000 LLM calls; latency table has a cell for every step and every attack type; plans unmodified; no `attacks.jsonl`; no BERTScore; no `experiments/reason/`; `attack_type_unresolved` = 0.
