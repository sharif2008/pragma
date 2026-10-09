# Validate: pragma-rag-eval-x (against SPEC)

Checklist is 1:1 with [`SPEC.md`](SPEC.md).

## Spec contract

- [ ] Action name `pragma-rag-eval-x`
- [ ] Flow input **only** from `experiments/data/eval-10|100|200/flows.csv` (default eval-200)
- [ ] Does not read `experiments/reason/` or old `fixtures/` for rows
- [ ] Two systems: A `LLM_only`, B `RAG_RANKING`. No `RAG_No_Ranking`
- [ ] Gold JSON N=90 (10×9) as class chunk concat; gold-100 **rows** not in the x-set
- [ ] Default x=200 → 400 LLM calls
- [ ] B retrieve: FAISS 80 + BM25 80 → RRF → MMR λ=0.5 → 20 → 5
- [ ] Headline: No RAG vs RAG + SHAP, BERTScore F1 / ROUGE-1 / ROUGE-L, **B−A**
- [ ] Per-attack proposed actions `k/n (p%)`
- [ ] Outputs under `experiments/rag/eval-x/`

## This run

- [ ] x lines in each jsonl
- [ ] `split_index` ∩ gold-100 = ∅
- [ ] `bertscore_rouge.png` + 9 action tables in `report.md`

## Verdict

**Spec/plan aligned to eval-x + `experiments/data/` input.** Re-check jsonl when the script writes `experiments/rag/eval-x/`.
