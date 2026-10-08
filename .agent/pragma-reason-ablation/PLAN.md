# Plan: all-type predict → RAG → rank → LLM plans + ablations

## Why

`detect-predict` now has one row per class, but `reason.py` (1) dumps mixed JSON under `experiments/reason/action_plans/`, (2) always pairs with-RAG / no-RAG only, (3) always runs the full rank cascade, (4) always injects SHAP into the query, (5) skips BENIGN, (6) records no per-step latency, (7) has no grouped README. The paper path in `docs/Pragma_v2.tex` §Reason needs a recorded 9×6 matrix through Mitigation Plans.

## Prerequisites (do not redo in this task)

- Checkpoint: `experiments/detect-train/vfl_model_best.pth`
- Index: `experiments/rag-index/vector_store/` + `rag_parents.json` (`pipeline.py rag-index`)
- Fixture: `experiments/detect-predict/inputs/all_attack_types.csv`
- `OPENAI_API_KEY` in `backend/.env`; model `gpt-4o-mini`

## Implementation order

1. **Forward pipeline extras (small)** — `pipeline.py` `run_stage` currently ignores `extra` except `e2e`. Forward `sys.argv` into `reason.py` / the new runner so `python scripts/pipeline.py reason -- --all-types-ablation` works.

2. **New runner** `backend/scripts/reason_ablation.py`
   - Load latest `predictions_detailed_*.json` from `experiments/detect-predict/` (run detect-predict first if missing).
   - Keep **one sample per `predicted_label`** (fixture is already 9 rows). Include BENIGN.
   - For each sample × six cells (SPEC table): build query → retrieve/rank as flagged → `create_agentic_orchestration_prompt` → `call_llm_api` ($T=0.3$).
   - Condition-off: class+confidence template; strip dominance sentence and top-feature JSON from the prompt (keep $W[\hat{y}_i]$ and domain names).
   - Ranking-off: after dense retrieve + merge, skip MMR and cross-encoder; take top children by vector score, then parent expand + budget 5.
   - RAG-off: skip FAISS; RAG slot = paper empty-KB sentence.
   - Time each step with `perf_counter` (ms). Copy detect ms into every cell of that sample.
   - Write `experiments/reason/all_types_<ts>/` exactly as SPEC. Generate README.md from the same JSON (do not hand-write).

3. **Reuse, do not duplicate**
   - Query: `reason.build_template_rag_query` + a class-only sister function.
   - Cascade: `retrieve_child_chunks_for_query`, `merge_and_dedupe_child_chunks`, `mmr_select`, `rerank_with_cross_encoder`, `expand_parent_sections`.
   - Prompt: `llm_prompt.create_agentic_orchestration_prompt` (paper Role + slots + JSON schema). Add a `include_conditions: bool` (or equivalent) rather than a second template.
   - Domain strings in query/README: Access / ISP, Perimeter / IDS, Endpoint / EDR (map RAN/Edge/Core).

4. **Leave `reason.py` main path alone** — still writes `action_plans/` with the existing with-RAG vs no-RAG pair. Ablation is the new script (or `--all-types-ablation` dispatch).

5. **Run**

   ```powershell
   cd D:\Projects\ChainAgentVFL\backend
   .\.venv\Scripts\Activate.ps1
   python scripts/pipeline.py detect-predict
   python scripts/reason_ablation.py
   ```

   Detect-predict is KernelSHAP-heavy (~9 rows). Ablation is 54 LLM calls; budget API cost/time. Optional `--classes DDOS,PORTSCAN` for a smoke run.

6. **Sanity on the paper cell** (`rag_cond_rank`)
   - Query contains class, confidence, a paper domain name, and feature keywords.
   - `rag_sections.json` has 1–5 parents with sources from the ten PDFs.
   - `plan.json` keys match Table plan-schema; every `action` ∈ $W[\hat{y}_i]$; every `network_tier` is one of the three domain strings.
   - `norag_*` README RAG slot is the empty-KB sentence, not invented citations.
   - `latency_summary.csv` has rows for predict, query, retrieve, rank (or skipped), prompt, llm.

## Done when

- `experiments/reason/all_types_<ts>/by_attack/` has **9** class folders, each with `prediction.json` and **6** cell folders.
- Each cell README shows prediction confidence, query, RAG (or empty-KB), plan, and latency.
- Paper cell uses SHAP-conditioned template query + full 7-step cascade + paper prompt.
- Root README lists the 9×6 matrix and total/mean ms per step.
- No writes under `backend/storage/`. No chain calls.
