# Validate: pragma-e2e-detect-chain (against SPEC)

Checklist is 1:1 with [`SPEC.md`](SPEC.md). Do not score this run with eval100 or agentic-attack VALIDATE.

## Spec contract

- [ ] Happy path only: detect → ranked RAG → reason → unmodified `storePlan` / verify / `markApplied`
- [ ] One system: **`RAG_RANKING`**. 1000 LLM calls. Resume key = `split_index`
- [ ] No plan mutation. No inject. No `attacks.jsonl` / `loops.jsonl`
- [ ] No BERTScore / ROUGE / evidence-support in `report.md`
- [ ] Retrieve via `rag_hybrid.hybrid_children` (FAISS 80 + BM25 80 → RRF → MMR λ=0.5 → 20 → 5). No new search. Does not read eval100 outputs
- [ ] Only experiment input: `experiments/data/e2e-detect-chain/`
- [ ] Does not import `agentic_attack_eval.py`
- [ ] All generated files under `experiments/e2e-detect-chain/`
- [ ] Does **not** write `experiments/reason/`
- [ ] `predicted_label` on every mitigation-plans case; `attackType` = detect ŷ
- [ ] `planId` = `{split_index}-RAG_RANKING-honest`
- [ ] Latency matrix in `report.md` **and** `.agent/pragma-e2e-detect-chain/REPORT.md`: columns Detect, Retrieve, Rank, LLM, Commit, Verify, Apply, E2E; rows Overall + 9 labels; every cell filled (`mean ms (n)`); no overall-only summary in place of this table
- [ ] `latency.json` has the same matrix (overall + 9 labels, each step mean and n)
- [ ] Job does not start or stop Hardhat

## This run

- [ ] Script `backend/scripts/reason_1000.py` exists
- [ ] Default command: sample (if missing) → detect → reason → chain → report
- [ ] `runs.jsonl` has 1000 `RAG_RANKING` lines
- [ ] `honest.jsonl` has 1000 lines; `attack_type_unresolved` = 0
- [ ] No `attacks.jsonl`, no inject flags
- [ ] `latency.json` has overall + 9 labels × each step
- [ ] `report.md` and `REPORT.md` contain the SPEC latency matrix (not a single overall mean)
- [ ] Other experiment folders not read or written

## Verdict

**Spec/plan aligned: happy path, self-contained, no `experiments/reason/`.** Execution not started.
