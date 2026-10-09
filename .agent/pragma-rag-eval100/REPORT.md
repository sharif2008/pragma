# rag_eval100 — A LLM vs B RAG+SHAP (SPEC)

Follows [`.agent/pragma-rag-eval100/SPEC.md`](SPEC.md). Not the old 90×3 three-system gold folder.

## Systems (SPEC)

| Paper | ID | Name | Prompt |
|-------|----|------|--------|
| A | `LLM_only` | LLM / No RAG | no RAG, no SHAP |
| B | `RAG_RANKING` | RAG + SHAP | 5 ranked parents + SHAP |

Do not run `RAG_No_Ranking`. Gold-100 traces stay in `experiments/rag/rag_eval100/`.

## This run

- **N = 200** flows from `rag_reason_500`, outside gold-100
- Mix: 23 BENIGN, 23 BOT, 22 DDOS / DOS / FTPPATATOR / OTHERS / PORTSCAN / SSHPATATOR / WEBATTACK
- **400** Mitigation Plans (`gpt-6-luna`, T=0)
- Folder: `experiments/rag/hybrid_eval200/`

B retrieve (SPEC): FAISS N=80 + BM25 N=80 → RRF → MMR λ=0.5 → 20 children → 5 parents → LLM.

## Score (SPEC)

Reference = gold-100 `relevant_rag_chunks[].text` concat **per attack type**.

| System | Hypothesis |
|--------|------------|
| A | rationale only |
| B | 5 parents + rationale |

Headline `bertscore_rouge.png`:

| Metric | No RAG (A) | RAG + SHAP (B) | B − A |
|--------|------------|----------------|-------|
| BERTScore F1 | *(after run)* | | |
| ROUGE-1 | | | |
| ROUGE-L | | | |

Proposed-action tables in `report.md`: each catalog slot `k/n (p%)` for A vs B, one section per attack type.

## Outputs (SPEC, flat)

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

Fill means and the nine action tables after 200×2 completes. Rebuild without new LLM calls:

```
python scripts/rag_eval100.py --offgold-n 200 --offgold-report-only --systems LLM_only,RAG_RANKING
```
