# Validate: pragma-e2e-detect-chain (against SPEC)

Checklist is 1:1 with [`SPEC.md`](SPEC.md). Do not score this run with eval100 or agentic-attack VALIDATE.

## Spec contract

- [ ] Happy path only: detect → ranked RAG → reason → unmodified `storePlan` / verify / `markApplied`
- [ ] One system: **`RAG_RANKING`**. 1000 LLM calls. Resume key = `split_index`
- [ ] No plan mutation. No inject. No `attacks.jsonl` / `loops.jsonl`
- [ ] No BERTScore / ROUGE / evidence-support in `report.md`
- [ ] Retrieve via `rag_hybrid.hybrid_children` (FAISS 80 + BM25 80 → RRF → MMR λ=0.5 → 20 → 5). No new search. Does not read eval100 outputs
- [ ] Experiment input is a frozen `flows.csv` (`--input`, default `experiments/data/e2e-detect-chain/`)
- [ ] Does not import `agentic_attack_eval.py`
- [ ] All generated files under `experiments/e2e-detect-chain/`
- [ ] Does **not** write `experiments/reason/`
- [ ] `predicted_label` on every mitigation-plans case; `attackType` = detect ŷ
- [ ] `planId` = `{split_index}-RAG_RANKING-honest`
- [ ] Latency matrix in `report.md` **and** `.agent/pragma-e2e-detect-chain/REPORT.md`: columns Detect, Retrieve, Rank, LLM, Commit, Verify, Apply, E2E; rows 9 labels then **Total last**; cells are total ms; E2E = row sum of the seven steps; Total = column sum of the nine attack types; no Total-only summary in place of this table
- [ ] `latency.json` has the same matrix (Total + 9 labels, each step sum/mean/min/max and n)
- [ ] Job does not start or stop Hardhat

## This run

- [x] Script `backend/scripts/e2e_detect_chain.py` exists
- [x] Default command: sample (if missing) → detect → reason → chain → report
- [ ] `runs.jsonl` has 1000 `RAG_RANKING` lines — **this run is eval-100 (N=100)**
- [x] `honest.jsonl` has 100 lines; `attack_type_unresolved` = 0
- [x] No `attacks.jsonl`, no inject flags
- [x] `latency.json` has Total + 9 labels × each step (sum/mean/min/max)
- [x] `report.md` and `REPORT.md` contain the SPEC latency matrix plus mean/min/max per step (not a single Total-only mean in place of the matrix)
- [x] Other experiment folders not read or written

## Verdict

**eval-100 complete (N=100, 100 LLM calls, unresolved=0).** Stored 99/100. One FTPPATATOR store revert (`87486`, `action_not_whitelisted`); job continued. Detect min=max (one amortized chunk). Full 1000-flow run not started.
