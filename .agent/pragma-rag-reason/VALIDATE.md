# Validate: pragma-rag-reason

## Checklist

- [ ] Fixture `rag_reason_500/flows.csv` exists; N=500; no gold overlap
- [ ] Default `python scripts/reason_500.py` runs reason then report
- [ ] Resume key is `(split_index, system)`; existing `RAG_RANKING` rows are kept
- [ ] `runs.jsonl` systems are `LLM_only`, `RAG_No_Ranking`, `RAG_RANKING`
- [ ] `report.md` is class-level only (`rate (k/n)`); no per-flow true/false table
- [ ] Phase columns: action correctness ↑, policy compliance ↑, unsafe ↓; winner underlined
- [ ] Per-attack table lists \(W[y]\) actions as percentages for all three systems
- [ ] Evidence-support / PDF needles are not headline metrics
- [ ] Chain / e2e not run

## Verdict

**Spec updated.** Default = A/B/C reason + report. Report is the paper comparison table, not a per-row dump.
