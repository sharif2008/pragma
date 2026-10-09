# RAG vs LLM-only (reserved gold cores)

**Run:** `experiments/rag/rag_eval100/`
**Gold:** `data/ground_truth/ground_truth-100.json` SHA-256 `7d9de2cff5f39b83826e967ec92ead1400a23f79afb6a25406c5cbaea5bd5388`
**Eval model:** `gpt-6-luna` (temperature 0)
**Index:** `D:\Projects\ChainAgentVFL\experiments\rag-index\vector_store`  ranking = MMR λ=0.5 then `rank_by_vector_score` (no cross-encoder)
**Flows:** `experiments/fixtures/gold-100/flows.csv` SHA-256 `c770442fa7eb768b8b41654ec6c66ff192f05486ba771a95951911f2e0ce598a`

## 1. What was compared

| ID | System | Retrieval |
|----|--------|-----------|
| A | LLM-only | none |
| B | Dense RAG | FAISS → parents |
| C | RAG + ranking | FAISS → MMR λ=0.5 k=60 → vector-score top-20 → 5 parents |

Same prompt, same $W[\hat{y}]$, **90 reserved cases (10 per class)**.

## 2. Gold

N=90 held-out VFL test rows (seed 42). Histogram 10 × 9 classes from `ground_truth-100.json`.
BERTScore / ROUGE reference = concatenation of `relevant_rag_chunks[].text`. IR labels = those `parent_id`s.

## 3. Results

Means below; one row per gold flow in `mitigation_plans.json` / `comparison.csv`.

| Metric | No RAG | RAG, no rank | RAG + rank |
|--------|-------:|-------------:|-----------:|
| Exact action | 0.256 | 0.200 | 0.189 |
| Acceptable action | 0.656 | 0.567 | 0.611 |
| BERTScore F1 (vs gold chunk concat) | 0.768 | 0.807 | 0.807 |
| ROUGE-1 | 0.139 | 0.282 | 0.282 |
| ROUGE-L | 0.071 | 0.116 | 0.116 |
| BERTScore F1 (retrieve-only, no LLM) | — | 0.807 | 0.807 |
| ROUGE-L (retrieve-only) | — | 0.116 | 0.116 |
| Recall@5 | — | 0.004 | 0.004 |
| MRR | — | 0.021 | 0.021 |

## 4. Why RAG context vs no RAG

Gold reference = concat of gold `relevant_rag_chunks[].text`. No-RAG hyp = rationale only; RAG hyp = retrieved parents + rationale.
Charts: `bertscore_rouge.png`, `rag_vs_norag.png`.

- RAG no-rank − no RAG (B−A): bertscore_f1 +0.039, rouge_1 +0.143, rouge_l +0.045, exact_action -0.056
- RAG+rank − no RAG (C−A): bertscore_f1 +0.039, rouge_1 +0.143, rouge_l +0.045, exact_action -0.067

A positive BERTScore / ROUGE delta means RAG system text is closer to the gold retrieved parents than no-RAG rationale.

## 5. Why ranking vs no ranking

- Rank − no rank (C−B): bertscore_f1 +0.000, rouge_1 +0.000, rouge_l +0.000, exact_action -0.011, recall@5 +0.000, mrr +0.000, retrieve_bertscore_f1 +0.000

A positive retrieve-only BERTScore means the context itself (no LLM) is closer to gold after MMR + vector-score rank.
If a delta is negative, ranking or RAG did not help on that metric — report the number.

## 6. Not this report

The 1000-row `pragma-e2e-detect-chain` set (later blockchain / e2e) is a different fixture.
No McNemar / Wilcoxon. No Commit / Apply.
