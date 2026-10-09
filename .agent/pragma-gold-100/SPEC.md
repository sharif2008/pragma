# Spec: reserve 10 test flows per class, then freeze gold

Action: `pragma-gold-100`

**N = 90** held-out flows: **10 per trained class** (9 labels). These rows are **reserved** for gold + `pragma-rag-eval100`. Do not reuse them in `pragma-rag-reason` (1000-row set) or for prompt tuning.

This task **stops at reserved flows + gold freeze**. It does not run LLM-only / RAG scoring.

Nine labels only (include BENIGN):

`BENIGN`, `BOT`, `DDOS`, `DOS`, `FTPPATATOR`, `OTHERS`, `PORTSCAN`, `SSHPATATOR`, `WEBATTACK`

No Heartbleed / Infiltration as their own class. All rows from the **VFL test split only** (`random_state=42`, 20% stratified). No train/val overlap.

## Shared input vs this task’s results

**Input** (owner `gold-100`; other tasks may read, must not write):

```
experiments/fixtures/gold-100/
  flows.csv          # 90 rows, 10 × 9 classes, seed 42
  manifest.json
  README.md
```

**Results** (this task only). Before each freeze, **delete leftover gold JSON** (`drafts/`, `gold_security_cases.*`, `generation.json`, `split_manifest.json`, `SHA256.txt`). **Keep** detect/SHAP: `predictions_*.json`, `predictions_*.csv`, `decision_summary_*.json`. Then overwrite the one gold file.

```
experiments/gold-100/
  predictions_detailed_*.json   # keep (SHAP / detect)
  ground_truth-100.json         # only gold output; overwrite
  README.md
```

Do not nest `gold_100/`. Do not write or archive `detect-predict/`. Do not emit extra gold JSON.

## Main output for `pragma-rag-eval100`

Eval reads **one file**:

`data/ground_truth/ground_truth-100.json`

Same bytes as `experiments/gold-100/ground_truth-100.json`. No CSV, no `generation.json`, no `SHA256.txt`, no drafts.

## Knowledge / actions (closed) — PDF-grounded gold

Gold is the **reference** that `pragma-rag-eval100` scores with the **LLM API** (`gpt-6-luna` / `OPENAI_API_KEY`). This task does **not** call that API. It must still write actions and rationales that a retrieved PDF can support.

- Corpus: ten PDFs via `resolve_rag_knowledge_dir()`; citations = `parent_id` from `rag_parents.json`
- **Select parents by body text**, not heading/preview. Heading OCR is often wrong (e.g. “Save this base image”). A parent is relevant only if its stored `text` supports the class or control (CIS 5/6/8/10/13/16/17, DoS, malware, reconnaissance, application security, incident logging).
- **Actions** stay verbatim `attack_options.json` → `W[true_label]`, but `primary_action` is the catalog string that the cited PDF control actually supports (not merely `W[0]`).
- Tiers: `Access / ISP` | `Perimeter / IDS` | `Endpoint / EDR` (prefer `primary_domains` ∩ SHAP-dominant).
- **`rationale`** quotes parent body text (plus true_label / SHAP). **`rationale_summary`** is a short 2–3 sentence explanation of the same decision.
- **`chunks_summary`** is a **semantic, human-readable digest of all kept parents** for that detection / RAG query (controls, safeguards, and what they require). It must **not** name PDFs, `parent_id`s, or filenames. `pragma-rag-eval100` uses it as the BERTScore / ROUGE reference against the LLM-generated rationale.
- Keep **full parent `text`** for the **top 20** matching parents (or all matches if fewer than 20). Store them in `relevant_rag_chunks[]` with `parent_id`, `source_file`, `section_heading`, pages, `text`, `text_hash`.
- If no parent text supports the case: `relevant_rag_chunks = []` and both rationale fields must say so. Empty is valid; a fake id is not.

Do not invent action names, PDFs, or chunk ids. Do not use `gpt-6-luna` here.

## Case schema (one object per reserved flow)

Each `cases[]` item stores detection, condition, rationale, and actions together:

- `detection`: `true_label`, `predicted_label`, `confidence`, SHAP (`dominant_domain`, `domain_shares`, `shap_features`)
- `condition`: `true_label`, `primary_network_tier`
- `rationale` (full, PDF-quoted), `rationale_summary` (short explanation), `chunks_summary` (semantic digest of retrieved chunks; BERTScore reference)
- `actions`: `primary_action`, `acceptable_actions`, `unsafe_actions`
- `relevant_rag_chunks[]`: up to 20 parents, **full `text`** each

Top-level also has `n`, `generation` (`openai_api_used: false`), `split` (seed, selected indices, class counts).

## Quality checks

- One file `ground_truth-100.json` in live and `data/ground_truth/` (identical)
- 90 unique `case_id` / `split_index`; histogram **10 each** of the nine classes
- Actions ∈ catalog; tiers ∈ three domains; cited parent body supports the class
- Detect/SHAP files remain under `experiments/gold-100/`

## Out of scope

- `pragma-rag-eval100` scoring
- Sampling the 1000-row `pragma-rag-reason` fixture
- Extra files in `data/ground_truth/`
- Stratified “majority BENIGN” 100-row draw
