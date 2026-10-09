# RAG retrieval scoring — A LLM vs B RAG+SHAP (SPEC)

Follows [`.agent/pragma-rag-eval100/SPEC.md`](SPEC.md). Not the old 90×3 three-system gold folder.

## Systems (SPEC)

| Paper | ID | Name | Prompt |
|-------|----|------|--------|
| A | `LLM_only` | LLM / No RAG | no RAG, no SHAP |
| B | `RAG_RANKING` | RAG + SHAP | 5 ranked parents + SHAP |

Do not run `RAG_No_Ranking`. Gold-100 traces stay in `experiments/rag/rag_eval100/`.

## This run

- **N = 200** flows from `rag_reason_500`, outside gold-100 (`split_index` overlap = 0)
- Mix: 23 BENIGN, 23 BOT, 22 DDOS / DOS / FTPPATATOR / OTHERS / PORTSCAN / SSHPATATOR / WEBATTACK
- **400** Mitigation Plans (`gpt-6-luna`, T=0)
- Folder: `experiments/rag/hybrid_eval200/`

B retrieve (SPEC): FAISS N=80 + BM25 N=80 → RRF → MMR λ=0.5 → 20 children → 5 parents → LLM.

Primary ∈ W[true]: No RAG **200/200** · RAG + SHAP **194/200**.

## Score (SPEC)

Reference = gold-100 `relevant_rag_chunks[].text` concat **per attack type**.

| System | Hypothesis |
|--------|------------|
| A | rationale only |
| B | 5 parents + rationale |

Headline `bertscore_rouge.png`:

| Metric | No RAG (A) | RAG + SHAP (B) | B − A |
|--------|------------|----------------|-------|
| BERTScore F1 | 0.774 | 0.817 | +0.043 |
| ROUGE-1 | 0.138 | 0.333 | +0.195 |
| ROUGE-L | 0.068 | 0.139 | +0.071 |

Proposed-action tables: `experiments/rag/hybrid_eval200/report.md` (nine `k/n (p%)` sections).

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

Rebuild without new LLM calls (uses existing jsonl; BERTScorer loads once):

```
python scripts/rag_retrieval_scoring.py --offgold-n 200 --offgold-report-only --systems LLM_only,RAG_RANKING
```
