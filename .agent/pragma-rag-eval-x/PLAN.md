# Plan: eval-x — follow SPEC (A LLM vs B RAG+SHAP)

Action: `pragma-rag-eval-x`. Source of truth: [`SPEC.md`](SPEC.md).

## Why

Compare No RAG vs ranked RAG+SHAP on **x** off-gold flows. Default x=200. Text baseline is gold chunk concat. No blockchain.

## Input

From `experiments/data/` only:

- Default 200: `experiments/data/eval-200/flows.csv`
- Also: `eval-10/`, `eval-100/`
- Gold JSON (score only): `experiments/gold-100/ground_truth-100.json`

Do not resample `rag_reason_500` at run time. Do not read `experiments/reason/` for flows or detect.

## Prerequisite

- Those three files
- `experiments/rag-index/vector_store/`
- `OPENAI_API_KEY` + `OPENAI_MODEL=gpt-6-luna`

## Steps

1. Verify gold JSON N=90 (10/class). Stop if not.
2. Read `experiments/data/eval-200/flows.csv` (or eval-10 / eval-100). Do not resample.
3. Detect into `experiments/rag/eval-200/detect/` if missing.
4. Mitigation Plans, T=0, resume jsonl:
   - A: no RAG, no SHAP
   - B: FAISS 80 + BM25 80 → RRF → MMR λ=0.5 → 20 children → 5 parents; SHAP on
5. Score vs class gold chunk concat. Write SPEC outputs.

```
python scripts/rag_retrieval_scoring.py --offgold-n 200 --systems LLM_only,RAG_RANKING
python scripts/rag_retrieval_scoring.py --offgold-n 200 --offgold-report-only --systems LLM_only,RAG_RANKING
```

## Done when

```
experiments/rag/eval-200/
  detect/predictions_detailed.json
  LLM_only.jsonl          # x lines
  RAG_RANKING.jsonl       # x lines
  mitigation_plans.json
  comparison.csv
  pipeline.png
  bertscore_rouge.png
  bertscore_rouge_3d.png
  bertscore_rouge_by_class_3d.png
  report.md
  manifest.json
```

x×2 plans; gold JSON untouched; flow CSV read only from `experiments/data/`.
