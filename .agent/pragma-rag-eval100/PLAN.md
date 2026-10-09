# Plan: eval100 — follow SPEC (A LLM vs B RAG+SHAP)

Source of truth: [`SPEC.md`](SPEC.md). This plan only sequences that spec.

## Why

Original eval100 contract: gold-100 is the text baseline, Mitigation Plans only, BERTScore/ROUGE vs chunk concat, no blockchain. This run is **200 off-gold flows** and **two systems**:

| Paper | ID | Name |
|-------|----|------|
| A | `LLM_only` | LLM / No RAG |
| B | `RAG_RANKING` | RAG + SHAP |

Do not run `RAG_No_Ranking`. Do not write `experiments/rag/rag_eval100/`.

## Prerequisite (SPEC)

- `data/ground_truth/ground_truth-100.json` — N=90, 10×9; class chunk concat
- `experiments/fixtures/rag_reason_500/flows.csv` + detect JSON
- `experiments/rag-index/vector_store/`
- `OPENAI_API_KEY` + `OPENAI_MODEL=gpt-6-luna`

## Steps

1. Verify gold N=90 (10/class). Stop if missing or not 10×9.
2. Sample 200 from reason_500, disjoint from gold `split_index`: **22/class +1 BENIGN +1 BOT**.
3. Mitigation Plans, T=0, resume jsonl in `experiments/rag/hybrid_eval200/`:
   - A: `include_knowledge_base=false`, `include_conditions=false`
   - B: hybrid FAISS N=80 + BM25 N=80 → RRF → MMR λ=0.5 → 20 children → 5 parents; SHAP on
4. Score vs **class** gold `relevant_rag_chunks[].text` concat. A = rationale; B = parents + rationale.
5. Write SPEC outputs: `mitigation_plans.json`, `comparison.csv`, `pipeline.png`, `bertscore_rouge.png` (No RAG vs RAG + SHAP, **B−A**), `report.md` with per-attack `k/n (p%)` tables.

```
python scripts/rag_eval100.py --offgold-n 200 --systems LLM_only,RAG_RANKING
python scripts/rag_eval100.py --offgold-n 200 --offgold-report-only --systems LLM_only,RAG_RANKING
```

## Done when (SPEC outputs)

```
experiments/rag/hybrid_eval200/
  LLM_only.jsonl
  RAG_RANKING.jsonl
  mitigation_plans.json
  comparison.csv
  pipeline.png
  bertscore_rouge.png
  report.md
  manifest.json
```

200×2 plans; gold-100 jsonl untouched; no RAG-only arm.
