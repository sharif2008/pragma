# Console demos (`backend/run/`)

Standalone scripts that call the running FastAPI backend over HTTP. Run everything from the **`backend/`** directory.

## Prerequisites

1. **Backend API** is up (e.g. `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`).
2. At least **one trained model** exists (Setting → train, or `POST /training/start`).
3. For **on-chain apply**: local Hardhat node, deployed `AgenticTrustRegistry`, whitelist seeded, and `TRUST_CHAIN_*` set in `.env` (see `hardhat-blockchain/README.md`).
4. Python deps installed (`pip install -r requirements.txt` — includes `httpx`).

Quick health check:

```bash
curl http://127.0.0.1:8000/health
```

---

## `attack_monitor.py` — CSV row-by-row pipeline + apply

Reads a VFL feature CSV **one row at a time**, runs prediction → RAG → agentic planning for each row, records the planned action set, then **applies actions** through the API (whitelist check + on-chain apply when configured).

**Defaults:** input `run/data/sample.csv`, **apply enabled**, **bulk** apply mode, output under `run/output/run_{timestamp}_{id}/`.

**Label rule:** every prediction is normalized to one key from `attack_options.json` (same catalog as `hardhat-blockchain/contracts/attack_options.json` / on-chain whitelist). Examples: `Web Attack – Brute Force` → `WEBATTACK`, `Infiltration` → `OTHERS`.

Each invocation creates one **flat** run folder (no per-row subfolders):

```
run/output/run_20250817_100530_a1b2c3d4/
  report.json         # summary totals + by_attack_class + chain_latency
  report.txt          # same summary, human-readable (includes latency)
  report.html         # attack-class dual-gate table + chain latency
  ledger.json         # full batch detail — all rows in one file
```

**`report.json`** fields:

| Field | Meaning |
|-------|---------|
| `total_rows` | CSV rows processed |
| `total_actions` | Planned / chain actions across all rows |
| `failed_actions` | Actions that did not apply successfully |
| `whitelist_fail_actions` | Actions rejected by smart-contract whitelist (Gate 1) |
| `labels_mapped_to_others` | Rows whose raw label is not in `attack_options.json` (mapped to `OTHERS`) |
| `by_attack_class` | Per-class proposed, changed, WL reject, plan reject, applied, block % of changed |
| `chain_latency` | Anchor store and getCommitment verify ms (per row, total, mean) |

### Run with defaults

```bash
cd backend

python run/attack_monitor.py --max-rows 1
```

Example with explicit CSV:

```bash
python run/attack_monitor.py --input-file run/data/sample_1000.csv --max-rows 3
```

### Apply phase (default)

When **`--apply`** is on (the default), after every CSV row completes the pipeline the script attempts to apply **all** detection actions for that row’s agentic report.

Use **`--apply-mode`** to choose how those actions are sent to the API:

| Mode | Default | Behavior |
|------|---------|----------|
| **`bulk`** | **yes** | One `POST /agent/reports/{id}/apply` per row — applies **all** actions in a single call |
| **`per-action`** | no | One `POST /agent/reports/{id}/apply-action` per action index |

With defaults (`--apply` + `--apply-mode bulk`), no extra flags are needed:

```bash
python run/attack_monitor.py
```

### Pipeline only (no apply)

```bash
python run/attack_monitor.py --no-apply
```

### First N rows from a specific file

```bash
python run/attack_monitor.py --input-file run/data/sample_1000.csv --max-rows 3
```

### Apply one action at a time

```bash
python run/attack_monitor.py --apply-mode per-action
```

### Tamper demo (optional, default off)

By default actions are applied **as planned** (matching the anchored report). Pass **`--tamper`** to rewrite **1 or 2** planned units per row (not the full plan) before apply. Rewrites mix two threats, drawn from `attack_options.json`:

- **Off-whitelist:** an action used by *other* attack classes, so Gate 1 (`isActionWhitelisted`) fails.
- **Plan-drift:** an action still on *this* class whitelist, but not in the committed plan, so Gate 2 (`action_plan_mismatch`) fails.

```bash
python run/attack_monitor.py --max-rows 1 --tamper
python run/attack_monitor.py --tamper --tamper-seed 42
```

The report tracks:

| Metric | Meaning |
|--------|---------|
| **Changed** | Substituted labels sent to apply (1–2 per row) |
| **WL reject** | Gate 1 — `action_not_whitelisted` |
| **Plan reject** | Gate 2 — `action_plan_mismatch` |
| **Block.% of changed** | (WL + plan rejects) / changed — expected **100** |
| **Tamper accepted** | Should stay **0** |

Apply order is **whitelist then plan-binding**. Restart the backend after updating so apply endpoints use that order.

`report.json` / `report.txt` / `report.html` include **anchor store** ms and **getCommitment verify** ms per row, plus run total and mean. There is no separate `latency.txt`.

### Batch report (printed + saved to run folder)

| Metric | Meaning |
|--------|---------|
| **Total rows** | CSV rows processed |
| **Total actions** | Planned / chain actions across all rows |
| **Failed actions** | Actions that did not apply (whitelist + integrity + other) |
| **Whitelist fail actions** | Actions rejected by smart-contract whitelist |
| **Applied OK** | Actions successfully applied on-chain |
| **Integrity fail** | Actions blocked due to modified or unauthorized plan |
| **Other fail** | Chain apply errors, HTTP errors, etc. |
| **Actions modified** | *(with `--tamper`)* Rewritten units sent to apply (off-whitelist and/or plan-drift) |
| **Tamper rejected (chain)** | *(with `--tamper`)* Modified actions blocked at Gate 1 or Gate 2 |
| **Tamper accepted (chain)** | *(with `--tamper`)* Rewrite that passed both gates (expected **0**) |

### All flags and defaults

| Flag | Default | Description |
|------|---------|-------------|
| `--input-file PATH` | `run/data/sample.csv` | CSV with VFL feature columns (optional `label` column) |
| `--base URL` | `http://127.0.0.1:8000` | API base URL |
| `--output-dir PATH` | `run/output/run_{timestamp}_{id}/` | Parent folder; each run gets a timestamped subfolder |
| `--start-row N` | `0` | Skip first N data rows after header |
| `--max-rows N` | *(all rows)* | Process at most N rows |
| `--latency-ms MS` | `0` | Simulated ingestion delay per row |
| `--force-error-step NAME` | *(empty)* | Inject pipeline failure at step name |
| `--timeout-s SEC` | `900` | HTTP timeout per request |
| `--pause-between-rows-s SEC` | `0` | Delay between pipeline rows |
| `--apply` / `--no-apply` | **`--apply`** | After pipeline rows finish, try to apply all detection actions (default: on) |
| `--apply-mode MODE` | **`bulk`** | How to apply: **`bulk`** = single apply call for all actions per row; **`per-action`** = one API call per action index |
| `--tamper` / `--no-tamper` | **`--no-tamper`** | Rewrite 1–2 units per row: mix off-whitelist (Gate 1) and plan-drift (Gate 2) |
| `--tamper-seed N` | `42` | RNG seed mixed with file_row for `--tamper` |
| `--pause-between-actions-s SEC` | `0.25` | Delay between per-action applies (only for `per-action` mode) |
| `--no-output-json` | off | Skip writing the run output folder |

### Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `No model_version found` | Train a model first |
| Apply skipped / no report | Pipeline row failed or no agentic job created |
| Whitelist fail | Action not in contract whitelist for that attack type — run `npm run seed:whitelist` in `hardhat-blockchain/` |
| Integrity fail | Trust anchor missing or plan modified — enable **Chain** when generating plans |
| HTTP timeout | Increase `--timeout-s`; LLM/RAG can be slow on first row |
| Input file not found | Use `run/data/sample.csv` or pass `--input-file run/data/sample_1000.csv` |

---

## `network_monitor.py` — single-row demo

One built-in sample row (or `--values-csv`), optional `--apply`. Modes: `simple`, `async-row`, `traffic`.

```bash
python run/network_monitor.py --mode simple

python run/network_monitor.py --mode simple --apply
```

See `python run/network_monitor.py --help` for all options.

---

## Idempotency

Each row uses `Idempotency-Key: {run_id}-row-{file_row}`. A new run id is generated on every invocation, so re-runs always create fresh pipeline runs and a new output folder.
