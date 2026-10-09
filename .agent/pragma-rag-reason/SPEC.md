# Spec: 500-row A/B/C mitigation scorecard (disjoint from gold-100)

Action: `pragma-rag-reason`

Paper row this run fills:

| Dimension | What we measure | Question |
|-----------|-----------------|----------|
| **Mitigation** | Catalog-legal actions; policy violations | Whether proposed responses stay in \(W[y]\) |

Gold-100 (`pragma-rag-eval100`) already scores 90 reserved cores against human gold. This action scores **500 other held-out test flows** — **never** those 90 `split_index` values.

Three systems on the **same** detect rows. **Three LLM calls per flow (1500 total)**. Policy PDFs do not name AEVRA actions or tiers — do not invent IR gold or score PDF lexical overlap as quality.

| ID | Report name | Retrieval |
|----|-------------|-----------|
| A `LLM_only` | No RAG | empty KB sentence |
| B `RAG_No_Ranking` | RAG | FAISS → merge → top-20 vector → 5 parents |
| C `RAG_RANKING` | Ranked RAG | FAISS → MMR → `rank_by_vector_score` → 5 parents |

## Fixture

```
experiments/fixtures/rag_reason_500/
  flows.csv
  manifest.json
  README.md
```

- Same training CSV, same `test_idx` (seed 42, 20% stratified)
- N = 500 after dropping every index in `fixtures/gold-100/manifest.json`
- Water-fill across nine labels (`OTHERS` is small — take all remaining)
- Blockchain / e2e **reuse this fixture**

Nine labels: BENIGN, BOT, DDOS, DOS, FTPPATATOR, OTHERS, PORTSCAN, SSHPATATOR, WEBATTACK.

## Scoring (internal; not printed per flow)

Group by **true_label**. Report class + overall rates. Underline the better system per phase (ties underlined together).

| Phase | ↑ / ↓ | True when |
|-------|-------|-----------|
| Action correctness | ↑ | parsed and primary action ∈ \(W[\text{true}]\) |
| Policy compliance | ↑ | parsed; every action ∈ \(W[\hat{y}]\); valid project tier |
| Unsafe action rate | ↓ | any action illegal for the true class |

Detect match (`predicted == true`) is a **context** column only (same detect for A/B/C).

Then, **per attack**, list every action in \(W[y]\). Cell = share of that class whose **primary** action is that slot.

## Report format (required)

Phase table (Overall, then each attack). Cells are `rate (k/n)`. Winner underlined.

```
| Phase | No RAG | RAG | Ranked RAG |
| Action correctness ↑ | … | <u>…</u> | … |
| Policy compliance ↑ | … | … | … |
| Unsafe action rate ↓ | … | … | … |
```

Action table per attack: `W[y]` rows × three systems.

No per-flow true/false listing.

## Run

Default: reason (resume per `(split_index, system)`) + report.

```
python scripts/reason_500.py
python scripts/reason_500.py --report-only
python scripts/reason_500.py --systems LLM_only
```

Sample / detect run only if the fixture or detect JSON is missing.

## Outputs

`experiments/reason/rag_reason_500/`

```
runs.jsonl
report.md
mitigation.json
summaries/mitigation.csv
coverage.json
manifest.json
README.md
```

## Out of scope

- Per-flow true/false tables in `report.md`
- Touching gold-100 / `data/ground_truth/`
- Evidence-support / PDF needle scores as headline metrics
- `storePlan` / Apply
