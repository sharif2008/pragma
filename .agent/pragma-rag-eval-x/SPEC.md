# Spec: RAG vs LLM-only (eval-x)

Action: `pragma-rag-eval-x` (formerly `pragma-rag-eval100`)

Script: `backend/scripts/rag_eval100.py`

**x** is the number of flows. Default **x = 200**. Two systems. Mitigation Plans only. No Commit / Apply.

| Paper | ID | Name |
|-------|----|------|
| **A** | `LLM_only` | LLM (no RAG, no SHAP) |
| **B** | `RAG_RANKING` | RAG + SHAP (hybrid retrieve + ranked parents + SHAP) |

Do **not** run `RAG_No_Ranking`.

## Input (from `experiments/data/` only)

Each test size has its **own** frozen CSV. Do not resample from `rag_reason_500` at run time. Do not read `experiments/reason/` or another task’s jsonl as the row source.

| x | Input | Results |
|--:|-------|---------|
| 10 | `experiments/data/eval-10/flows.csv` | `experiments/rag/eval-10/` |
| 100 | `experiments/data/eval-100/flows.csv` | `experiments/rag/eval-100/` |
| **200** (default) | `experiments/data/eval-200/flows.csv` | `experiments/rag/eval-200/` |

`eval-10` ⊂ `eval-100` ⊂ `eval-200`. All disjoint from `experiments/data/gold-100/`.

Detect for those rows is produced by **this** task into `experiments/rag/eval-x/detect/` from that CSV.

### Gold text baseline (scoring only, not the flow CSV)

BERTScore / ROUGE still need the frozen chunk concat:

`experiments/gold-100/ground_truth-100.json`. **N = 90** (10 × 9). Stop if missing or not 10×9. Do not edit gold. Do not use gold-100 **rows** as the x-flow set.

| Field | Used as |
|-------|---------|
| `relevant_rag_chunks[].text` **concatenated per attack type** | BERTScore + ROUGE reference |
| `actions.*` | gold exact / acceptable / unsafe (optional side tables) |

## LLM

- Model: `OPENAI_MODEL` (default **`gpt-6-luna`**), T=0
- Prompt: `create_agentic_orchestration_prompt` — only flags differ by system
- \(W[\hat{y}]\) from `attack_options.json`
- **x × 2** Mitigation Plans; resume jsonl

```
python scripts/rag_eval100.py --offgold-n 200 --systems LLM_only,RAG_RANKING
```

## Systems

| ID | Name | RAG in prompt | SHAP in prompt | Retrieve |
|----|------|---------------|----------------|----------|
| A `LLM_only` | LLM / No RAG | no | no | none |
| B `RAG_RANKING` | RAG + SHAP | yes (5 ranked parents) | yes | hybrid + MMR λ=0.5 |

Index: `experiments/rag-index/vector_store/`. Query: `build_template_rag_query`.

### Retrieve cascade (B only)

1. Dense FAISS — N=80
2. BM25 — N=80, same child ids
3. RRF — \(1/(60+\mathrm{rank})\)
4. MMR — \(\lambda=0.5\), keep **20** children
5. **5** parents into the LLM prompt

## What to score

**BERTScore / ROUGE:** class-level concat of gold `relevant_rag_chunks[].text`. A = rationale; B = prompt parents + rationale.

Headline `bertscore_rouge.png`:

| Metric | No RAG (A) | RAG + SHAP (B) | B − A |
|--------|------------|----------------|-------|
| BERTScore F1 | … | … | … |
| ROUGE-1 | … | … | … |
| ROUGE-L | … | … | … |

Per-attack proposed-action tables: `k/n (p%)` from Mitigation Plans. Underline the higher share when > 0.

## Outputs

`experiments/rag/eval-200/` (default). Smoke and mid: `eval-10/`, `eval-100/`.

```
detect/predictions_detailed.json
LLM_only.jsonl
RAG_RANKING.jsonl
mitigation_plans.json
comparison.csv
pipeline.png
bertscore_rouge.png
report.md
manifest.json
```

## Out of scope

- `RAG_No_Ranking`
- Gold-100 90×3 re-run; using gold-100 **rows** as the x-set
- `pragma-e2e-detect-chain` / blockchain / Apply
- Reading flow CSVs from anywhere except `experiments/data/`
