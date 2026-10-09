# Spec: 1000-flow happy-path e2e latency

Action: `pragma-e2e-detect-chain`

Script: `backend/scripts/e2e_detect_chain.py`

Self-contained success path on **1000** flows. One system. One LLM call per flow. **No plan mutation. No inject. No BERTScore.**

**detect → ranked RAG → reason → `storePlan` → verify → `markApplied`**

This run does **not** open any other experiment folder, report, gold JSON, or mitigation-plans file. It does not score eval100 or agentic-attack. It does not edit other `.agent` tasks.

## What we measure

| Dimension | Clock | Question |
|-----------|--------|----------|
| **E2E latency** | Detect, retrieve, rank, LLM, commit, verify, apply | How long does an unmodified success-path flow take, per attack type? |
| **Honest chain** | `storePlan` / `getPlan` / `markApplied` | Did the generated plan land as written? |

## System (one only)

| ID | Name | LLM calls |
|----|------|-----------|
| `RAG_RANKING` | Ranked RAG + SHAP | **1000** |

Do **not** run `LLM_only` or `RAG_No_Ranking`. Store the LLM plan **as generated**. Do not swap actions, tiers, or prose before commit.

### Retrieve (library, not another task)

Call `scripts.rag_hybrid.hybrid_children` + parent expand (`rag_bridge.retrieve_context` / `reason.expand_parent_sections`). Do **not** read `experiments/rag/**` jsonl or eval100 reports. Do **not** add a new search path.

Index on disk: `experiments/rag-index/vector_store/` (live FAISS, not an eval artifact). Query: `build_template_rag_query`.

1. Dense FAISS — N=80
2. BM25 — N=80, same child ids
3. RRF — \(1/(60+\mathrm{rank})\)
4. MMR — \(\lambda=0.5\), keep **20** children
5. **5** parents into the prompt

Prompt: `create_agentic_orchestration_prompt` with `include_knowledge_base=True`, `include_conditions=True`. Model `OPENAI_MODEL` (default `gpt-6-luna`), temperature **0**. \(W[\hat{y}]\) from `attack_options.json` using the **detect** label.

## Input (this task only)

Default fixture:

```
experiments/data/e2e-detect-chain/
  flows.csv
  manifest.json
  README.md
```

`--input PATH` may point at another frozen `experiments/data/<set>/` folder or `flows.csv` (e.g. `eval-10`). Do not resample. N is that CSV’s length.

- Default N = **1000** from the VFL test split (seed **42**, 20% stratified) of the training CSV
- Water-fill across nine labels (`OTHERS` is small — take all remaining)
- Confirm histogram in **that** fixture’s `manifest.json`
- Do not read `experiments/gold-100/ground_truth-100.json` or any other task’s jsonl/report

Nine labels: BENIGN, BOT, DDOS, DOS, FTPPATATOR, OTHERS, PORTSCAN, SSHPATATOR, WEBATTACK.

Results never go in `experiments/data/`.

## Pipeline

Default = full happy path with resume. Do not inject, mutate, or re-reason.

| Stage | Job | Resume |
|-------|-----|--------|
| **Sample** | Write this fixture | skip if `flows.csv` exists and N=1000 |
| **Detect** | VFL + SHAP on this fixture | `detect/predictions_detailed.json` (chunked `detect_predict.py` into **this** live folder) |
| **RAG + reason** | Ranked retrieve + one Mitigation Plan | `runs.jsonl` key = `split_index` |
| **Chain** | Unmodified `storePlan` → `getPlan`/`isInPlan` → `markApplied` | `honest.jsonl` key = `split_index` |
| **Report** | Latency + honest store, per attack type | `report.md` from this folder’s jsonl |

### Chain (happy path only)

For each of the 1000 plans, **no modification**:

1. `attackType` = detect `predicted_label`. Never substitute `true_label`. Write `predicted_label` on the case and the system block.
2. `planId` = `{split_index}-RAG_RANKING-honest`
3. `storePlan` the generated units as written → `getPlan` / `isInPlan` → `markApplied` each `{u,τ}`
4. On revert, record and **continue**. Do not rewrite the plan and retry.

Use `trust_chain_service` (or equivalent JSON-RPC). Do **not** import `agentic_attack_eval.py`. Do **not** call C1–C6, T4–T7, L1/L2, or Auth.

Hardhat is already up (`http://127.0.0.1:8545`, chain 31337). This job does not start or stop it.

`attack_type_unresolved` on honest store must be **0**.

## What to score

Group by **true_label**, then Total. Latency cells are **total ms** (sum over that row’s n flows).

**Row-wise:** E2E = Detect + Retrieve + Rank + LLM + Commit + Verify + Apply. **Column-wise:** Total n and each Total step = sum of the nine attack-type rows. Honest-store Total n / Stored / Applied / Fail / unresolved are those same column sums.

### Latency (required table)

One matrix: **row = attack type**, **column = pipeline step**. Nine labels first, **Total last**. This table is the headline of `report.md` and `.agent/pragma-e2e-detect-chain/REPORT.md`.

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | | | | | | | | | |
| BOT | | | | | | | | | |
| DDOS | | | | | | | | | |
| DOS | | | | | | | | | |
| FTPPATATOR | | | | | | | | | |
| OTHERS | | | | | | | | | |
| PORTSCAN | | | | | | | | | |
| SSHPATATOR | | | | | | | | | |
| WEBATTACK | | | | | | | | | |
| Total | 1000 | | | | | | | | |

| Column | Clock |
|--------|--------|
| Detect | wall per chunk, amortized per flow |
| Retrieve | FAISS + BM25 + RRF |
| Rank | MMR |
| LLM | Mitigation Plan |
| Commit | `storePlan` |
| Verify | `getPlan` / `isInPlan` |
| Apply | honest `markApplied` total per attack type |
| E2E | row sum of the seven steps |

`latency.json` stores the same matrix (each label + Total; each step’s `sum_ms`, `mean_ms`, `min_ms`, `max_ms`, and n).

### Honest store

Same row set (9 labels, then Total):

| Attack | n | Stored | Applied | Fail | unresolved |
|--------|--:|-------:|--------:|-----:|-----------:|
| … | | | | | |
| Total | 1000 | | | | **0** |

`attack_type_unresolved` must be **0**.

No BERTScore, ROUGE, evidence-support, inject Block.%, or per-flow true/false action tables.

## Report format

`experiments/e2e-detect-chain/report.md` (filled after the run) and `.agent/pragma-e2e-detect-chain/REPORT.md` (same tables). Order:

1. Fixture N, class histogram, LLM calls = 1000
2. **Latency matrix** (required): every step × every attack type, Total last (cells = total ms)
3. **E2E per-flow** mean / min / max (9 labels, then Total)
4. **Total by step** mean / min / max / sum
5. Honest store matrix (9 labels, then Total)
6. Chain fails (if any), by `split_index` and revert

Do not replace the latency matrix with a single overall mean. Every step and every attack type must have a cell. Mean / min / max sit in extra tables, not instead of the totals.

## Run

```
cd backend
python scripts/e2e_detect_chain.py
python scripts/e2e_detect_chain.py --input ../experiments/data/eval-10
python scripts/e2e_detect_chain.py --sample-only
python scripts/e2e_detect_chain.py --predict-only
python scripts/e2e_detect_chain.py --reason-only
python scripts/e2e_detect_chain.py --chain-only
python scripts/e2e_detect_chain.py --report-only
```

Default = sample (if missing) → detect (if missing) → reason (resume) → chain (resume) → report.

`--systems` is not accepted.

## Outputs (one folder)

`experiments/e2e-detect-chain/`

```
detect/predictions_detailed.json
runs.jsonl
mitigation_plans.json
honest.jsonl
latency.json               # Total + 9 labels × each step
report.md                  # latency matrix required
manifest.json
README.md
```

Also fill `.agent/pragma-e2e-detect-chain/REPORT.md` with the same latency matrix.

## Out of scope

- Plan mutation, inject (C1–C6, T4–T7), loops, Auth B1–B8
- BERTScore / ROUGE / PDF needles / evidence-support
- `LLM_only` / `RAG_No_Ranking`
- Reading or writing other experiment folders (`reason/`, `gold-100/`, `rag/**`, `agentic-attack/`, `e2e/`)
- Other-task reports as inputs
- Editing other `.agent` tasks or `docs/Pragma_v2.tex`
- Starting Hardhat; live SOAR / MCP
- A new search implementation
