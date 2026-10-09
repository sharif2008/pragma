# Validate: pragma-rag-eval100 (against SPEC)

Checklist is 1:1 with [`SPEC.md`](SPEC.md). Gold-100 is read-only. This run is A vs B, n=200.

## Spec contract

- [x] Two systems only: **A `LLM_only`**, **B `RAG_RANKING` (RAG + SHAP)**
- [x] Old middle system `RAG_No_Ranking` not in this run
- [x] Gold-100 N=90 (10×9) required as class chunk concat; do not edit; do not write `rag_eval100/`
- [x] LLM = `gpt-6-luna`, T=0, $W[\hat{y}]$, Mitigation Plans only
- [x] 200 × 2 = 400 calls; resume jsonl
- [x] B retrieve: FAISS 80 + BM25 80 → RRF → MMR λ=0.5 → 20 children → 5 parents
- [x] BERTScore/ROUGE vs class gold chunk concat (A = rationale, B = parents + rationale)
- [x] Headline chart: No RAG vs RAG + SHAP, BERTScore F1 / ROUGE-1 / ROUGE-L, **B−A**
- [x] Per-attack proposed actions: `k/n (p%)`, underline winner
- [x] Out of scope: gold 90×3 re-run, reason-1000, e2e, inventing IR gold

## This run

- [x] Sample mix 23 BENIGN, 23 BOT, 22 each other class (from live picker)
- [ ] `split_index` ∩ gold-100 = ∅ (enforced in `load_offgold_pairs`)
- [ ] 200 lines in `hybrid_eval200/LLM_only.jsonl`
- [ ] 200 lines in `hybrid_eval200/RAG_RANKING.jsonl`
- [ ] `mitigation_plans.json` (A + B full primary + supporting)
- [ ] `pipeline.png`
- [ ] `bertscore_rouge.png` title `No RAG vs RAG + SHAP`, B−A labels
- [ ] `report.md` has 9 action tables
- [x] `experiments/rag/rag_eval100/*.jsonl` not written by this run

## Verdict

**Spec/plan aligned.** Execution in progress (A `LLM_only` first). Re-check jsonl counts and artifacts when 400 calls finish.
