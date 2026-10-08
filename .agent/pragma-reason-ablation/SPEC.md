# Spec: all-type samples → predict → RAG query → rank → LLM action plans

Action: `pragma-reason-ablation`

Run the paper **Reasoning and Planning Layer** (`docs/Pragma_v2.tex` §Reason, Fig. rag-retrieval + prompt-construct) on **one fixture row per trained class**. Stop at Mitigation Plans. Do **not** Commit or Apply.

Each attack folder must show, in order: prediction (label + confidence + SHAP), the filled RAG query, retrieve-then-rank context, the paper prompt + JSON response, and per-step latency. Then repeat the same row under ablation switches (RAG / condition / ranking).

## Paper path (evaluated cell)

`docs/Pragma_v2.tex` §Reasoning and Planning Layer. Deterministic; no LLM-written search string.

1. **Detect product** — $\hat{y}_i$, $\mathrm{conf}_i$, dominant domain $p_i^{\star}$, three domain shares, top-3 features per domain, top-5 $|\phi|$ overall. Domains in every later string: **Access / ISP**, **Perimeter / IDS**, **Endpoint / EDR**.
2. **Query** — one template filled from that record (class, confidence, dominant domain, three keyword lists). Optional rephrases exist in code; this task uses **template only**.
3. **Index** — closed corpus already built by `rag-index` (child 384 / overlap 96 / MiniLM; ranking on children; prompt gets parents).
4. **Retrieve-then-rank cascade** (seven steps): dense retrieve → merge/dedupe → MMR ($\lambda=0.5$, $k=60$) → cross-encoder `ms-marco-MiniLM-L-6-v2` → top-20 children → parent expand (12k cap) → prompt budget **5** sections (up to 10). Empty list → explicit “no relevant documents” sentence; model must not invent policy.
5. **Prompt** — Role (fixed) + runtime slots: (i) three domain strings, (ii) prediction summary, (iii) dominance sentence, (iv) evidence JSON top-5 features, (v) $W[\hat{y}_i]$, (vi) per-domain cards, (vii) numbered RAG sections or empty-KB sentence, (viii) task block. GPT-4o-mini, $T=0.3$. First `{` … last `}` parsed.
6. **Response** — one `plan_i` object (Table plan-schema): `threat_level`, `all_actions`, `primary_actions[]`, `supporting_actions[]`, `overall_reasoning`, `execution_priority`, `knowledge_sources_used`. `action` copied from $W[\hat{y}_i]$ (no paraphrase). `network_tier` exactly one of the three domain strings.

## Inputs (read, do not copy)

| Source | Role |
|--------|------|
| `experiments/detect-predict/inputs/all_attack_types.csv` | 9 rows, one per class (`resolve_sample_csv()` already prefers this) |
| `experiments/detect-train/run_*/` | **Latest** live train run (`resolve_model_dir()`); older runs under `archive/` |
| `experiments/detect-predict/run_*/` | **Latest** live predict run (`resolve_latest_predict_dir()`); older runs under `archive/` |
| `experiments/rag-index/vector_store/` | FAISS + `rag_parents.json` |
| `backend/storage/attack_options.json` | $W[\hat{y}_i]$ |
| `backend/storage/agentic_features.json` | domain cards |

Fallback CSV: `backend/run/data/sample_all_attack_types.csv`.

Classes (include **BENIGN**; do not skip it): BENIGN, BOT, DDOS, DOS, FTPPATATOR, OTHERS, PORTSCAN, SSHPATATOR, WEBATTACK.

## Ablation matrix (per sample)

Paper prompt + JSON schema are **on for every cell**. Ranking is skipped when RAG is off.

| Cell id | RAG | Condition (SHAP in query + prompt) | Ranking (MMR + cross-encoder) |
|---------|-----|-------------------------------------|-------------------------------|
| `rag_cond_rank` | on | on | on — **evaluated / paper path** |
| `rag_cond_norank` | on | on | off — dense top-$k$ children → parents |
| `rag_nocond_rank` | on | off | on |
| `rag_nocond_norank` | on | off | off |
| `norag_cond` | off | on | n/a — empty-KB sentence in RAG slot |
| `norag_nocond` | off | off | n/a |

**Condition on:** query and prompt include $p_i^{\star}$, domain shares, top-3 per domain, top-5 $|\phi|$.  
**Condition off:** query is class + confidence only (no feature keywords / dominance); prompt evidence JSON and dominance sentence are omitted or replaced with “(no SHAP conditions)”. $W[\hat{y}_i]$ and domain names stay.

Optional extra cell (flag `--strip-prompt`): `rag_cond_rank` again with Role/slots/schema stripped to a free-form “write a mitigation” user message. Default run does **not** include it.

$9$ samples $\times$ $6$ cells $= 54$ LLM calls.

## Pipeline list (one sample, one cell)

1. **Predict** (once per row, shared across cells) — VFL + KernelSHAP; write label, confidence, dominant domain, shares, feature $\phi$.
2. **Build query** — template fill (conditioned or not).
3. **Dense retrieve** — if RAG on.
4. **Rank cascade** — if RAG on and ranking on; else dense cut only.
5. **Assemble prompt** — paper slots; RAG slot is numbered parents or empty-KB sentence.
6. **LLM decode** — GPT-4o-mini $T=0.3$; parse JSON plan.

Wall-clock **ms** for each numbered step (and LLM token usage if the API returns it). Predict time may be amortized across cells (record once, copy into each cell README).

## Output layout

All writes under `experiments/reason/all_types_<YYYYmmdd_HHMMSS>/` (no new `experiment_dir` task; do not write `backend/storage/`).

```
experiments/reason/all_types_<ts>/
  README.md                 # run config, 9×6 matrix, latency totals
  latency_summary.csv       # sample, cell, step, ms
  latency_summary.json
  by_attack/
    <CLASS>/                # e.g. DDOS/
      README.md             # prediction + one table of six cells
      prediction.json       # detect product (shared)
      <cell_id>/
        README.md           # query, RAG titles, plan summary, latency
        query.txt
        rag_sections.json   # [] when no RAG
        prompt.txt
        response_raw.txt
        plan.json           # parsed plan_i (or parse_error)
        latency.json
```

Root and per-class `README.md` must be human-readable and include:

- predicted label, confidence, true label if present, dominant domain + %, three shares
- the exact query string
- ranked section titles + sources (or empty-KB)
- plan: threat_level, execution_priority, primary/supporting `{action, network_tier}`
- latency table for that cell / class

## Script / launch

New runner: `backend/scripts/reason_ablation.py`. Reuse `reason.py` retrieve/MMR/cross-encoder and `llm_prompt.py` slot filler; do not fork a second prompt template.

```powershell
cd D:\Projects\ChainAgentVFL\backend
.\.venv\Scripts\Activate.ps1
python scripts/pipeline.py detect-predict
python scripts/reason_ablation.py
```

After `pipeline.py` forwards `extra` for non-e2e stages, this is also valid:

```powershell
python scripts/pipeline.py reason -- --all-types-ablation
```

`reason.py` default (dumps into `action_plans/` without per-class folders) stays unchanged.

## Out of scope

- `storePlan` / Apply / Hardhat / `e2e`
- FastAPI `/agent/decide` and `backend/storage/vector_db`
- Retrain, rebuild FAISS (prereq: `rag-index` already built)
- LLM-authored retrieval queries (`llm_concise` / `llm_expanded`)
