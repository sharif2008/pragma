# Spec: RAG vs LLM-only on gold-100 ground truth

Action: `pragma-rag-eval100`

Script: `backend/scripts/rag_eval100.py`

Two systems only:

| Paper | ID | Name |
|-------|----|------|
| **A** | `LLM_only` | LLM (no RAG, no SHAP) |
| **B** | `RAG_RANKING` | RAG + SHAP (hybrid retrieve + ranked parents + SHAP) |

Do **not** run the old middle system (`RAG_No_Ranking`). Do **not** overwrite gold-100 traces.

For **each selected flow**, call the LLM (`gpt-6-luna`) and store a Mitigation Plan. Gold freeze does **not** call this API. Stop at Mitigation Plans. Do not Commit or Apply.

## Ground truth (read-only)

`data/ground_truth/ground_truth-100.json` (same bytes as `experiments/gold-100/ground_truth-100.json`). **N = 90** (10 per class × 9 labels). Filename keeps `-100`; do not resample.

| Field | Used as |
|-------|---------|
| `detection` | attach detect + SHAP + confidence |
| `condition.true_label` / `primary_network_tier` | class histogram; optional tier match |
| `actions.primary_action` / `acceptable_actions` / `unsafe_actions` | gold-100 exact / acceptable / unsafe only |
| `chunks_summary` | human digest (not the BERTScore/ROUGE baseline) |
| `relevant_rag_chunks[].text` **concatenated** | **BERTScore + ROUGE baseline** — one concat **per attack type** for off-gold |
| `relevant_rag_chunks[].parent_id` | IR relevant set on gold-100 only |

Flows for this run: 200 rows from `experiments/fixtures/rag_reason_500/flows.csv` (detect in `experiments/reason/rag_reason_500/`). **Outside** gold-100 `split_index`. Do not edit gold. Missing gold or not 10×9 → **stop** (gold file still required as the class baseline).

**Gold-100 is the human quality bench.** This 200-flow run does not invent IR gold and does not write `experiments/rag/rag_eval100/`.

## LLM (required)

- Model: `OPENAI_MODEL` (default **`gpt-6-luna`**)
- Key: `OPENAI_API_KEY` in `backend/.env`
- Prompt: `create_agentic_orchestration_prompt` — only flags differ by system
- $W[\hat{y}]$ from `attack_options.json` (predicted label)
- Temperature **0**; invalid JSON → `format_error=true`
- **200 flows × 2 systems** = **400** Mitigation Plans; resume jsonl at the run folder

```
python scripts/rag_eval100.py --offgold-n 200 --systems LLM_only,RAG_RANKING
```

## Systems

Same 200 flows, same model. Catalog $W[\hat{y}]$ stays in both.

| ID | Name | RAG in prompt | SHAP in prompt | Retrieve |
|----|------|---------------|----------------|----------|
| A `LLM_only` | LLM / No RAG | no | no | none |
| B `RAG_RANKING` | RAG + SHAP | yes (5 ranked parents) | yes | hybrid + MMR λ=0.5 |

`include_knowledge_base` = RAG on/off. `include_conditions` = SHAP on/off.

Index: `experiments/rag-index/vector_store/`. Template query (`build_template_rag_query`).

### Retrieve cascade (system B only)

Same child ids for both channels.

1. **Dense FAISS** — N=80, score $=1/(1+d)$
2. **BM25** — N=80, same FAISS child texts / ids
3. **RRF** — $1/(60+\mathrm{rank})$
4. **MMR** — $\lambda=0.5$, keep **20** children
5. **Parents** — expand → **5** parents into the LLM prompt

## This run (200 flows)

Stratified from `rag_reason_500`, disjoint from gold-100:

- **22 per class × 9 = 198**, remainder **+1 BENIGN, +1 BOT** → **200**
- Writes **`experiments/rag/hybrid_eval200/`** only
- Smoke n=10 stays in `experiments/rag/hybrid_smoke10/`

## What to score

**BERTScore / ROUGE reference:** class-level concat of gold `relevant_rag_chunks[].text` (same blob for every flow of that attack type). Not `chunks_summary`.

| System | Scored text |
|--------|-------------|
| A LLM | Mitigation Plan rationale only |
| B RAG + SHAP | retrieved prompt parents **then** rationale |

**Headline chart** `bertscore_rouge.png` — **only** this table:

| Metric | No RAG (A) | RAG + SHAP (B) | B − A |
|--------|------------|----------------|-------|
| BERTScore F1 | … | … | … |
| ROUGE-1 | … | … | … |
| ROUGE-L | … | … | … |

Figure: grouped bars, values on bars, **B−A** above each pair. Title: `No RAG vs RAG + SHAP`. Helper: `write_ac_text_chart`.

**Proposed-action tables** (from Mitigation Plans, not gold exact). An action counts if it is in `primary_actions` or `supporting_actions`. Cell = `k/n (p%)`. Underline the higher share when > 0.

```
### DDOS  n=22  W[y] = limit rate, enable scrubbing, …

| Action | No RAG | RAG + SHAP |
| limit rate | 18/22 (81.8%) | 20/22 (90.9%) |
```

Do not import `evaluate.py`. BERTScore + `rouge_score` live in `rag_eval100.py`.

## Outputs (flat)

`experiments/rag/hybrid_eval200/`:

```
LLM_only.jsonl
RAG_RANKING.jsonl
mitigation_plans.json
comparison.csv
pipeline.png
bertscore_rouge.png
report.md
manifest.json
```

`pipeline.png` = FAISS + BM25 → RRF → MMR → 20 children → 5 parents → LLM (B). A skips retrieve.

## Out of scope

- Old RAG-only (no SHAP / no rank) system
- Re-running gold-100 90×3
- 1000-row `pragma-rag-reason` / blockchain / Apply / e2e
- Editing gold; reindex
- Using gold-100 rows as the 200-flow set
