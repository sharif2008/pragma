# E2E detect → RAG → reason → chain — latency

Follows [`SPEC.md`](SPEC.md). Fill cells after `reason_1000.py` runs. Live copy: `experiments/e2e-detect-chain/report.md`.

Happy path only. Plans stored as generated. No inject. No BERTScore.

- Input: `experiments/data/e2e-detect-chain/` (N=1000)
- System: `RAG_RANKING` (1000 LLM calls)
- Output: `experiments/e2e-detect-chain/`

## Latency (mean ms)

Row = attack type. Column = pipeline step. Cell = `mean ms (n)`.

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| Overall | 1000 | | | | | | | | |
| BENIGN | | | | | | | | | |
| BOT | | | | | | | | | |
| DDOS | | | | | | | | | |
| DOS | | | | | | | | | |
| FTPPATATOR | | | | | | | | | |
| OTHERS | | | | | | | | | |
| PORTSCAN | | | | | | | | | |
| SSHPATATOR | | | | | | | | | |
| WEBATTACK | | | | | | | | | |

| Step | Clock |
|------|--------|
| Detect | VFL + SHAP, amortized per flow |
| Retrieve | FAISS + BM25 + RRF |
| Rank | MMR |
| LLM | Mitigation Plan |
| Commit | `storePlan` |
| Verify | `getPlan` / `isInPlan` |
| Apply | honest `markApplied` (per plan) |
| E2E | sum of the seven steps |

## Honest store

| Attack | n | Stored | Applied | Fail | unresolved |
|--------|--:|-------:|--------:|-----:|-----------:|
| Overall | 1000 | | | | **0** |
| BENIGN | | | | | |
| BOT | | | | | |
| DDOS | | | | | |
| DOS | | | | | |
| FTPPATATOR | | | | | |
| OTHERS | | | | | |
| PORTSCAN | | | | | |
| SSHPATATOR | | | | | |
| WEBATTACK | | | | | |

Chain fails (if any) are listed in the live `report.md` by `split_index` and revert.
