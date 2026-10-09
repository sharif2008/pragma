# Plan: 500 disjoint flows → A/B/C reason + report

## Why

Paper **Mitigation** row on 500 held-out flows that are not gold-100. Compare No RAG vs RAG vs Ranked RAG. Score the closed catalog, not PDF wording.

## Default

```
python scripts/reason_500.py
```

Runs **reason** (resume `runs.jsonl` per system) then **report**. Sample / detect only if missing.

`--report-only` rebuilds the tables from existing runs. `--systems` can run a subset.

## Report

Per attack and overall: phase table (action correctness ↑, policy ↑, unsafe ↓) with the better system underlined. Then \(W[y]\) action shares as percentages.

## Done when

- Fixture N=500, zero overlap with gold
- `runs.jsonl` has up to 1500 rows (`LLM_only` + `RAG_No_Ranking` + `RAG_RANKING`)
- `report.md` has phase winners + per-attack action % tables
- No writes under `data/ground_truth/` or `rag_eval100/`
