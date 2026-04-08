# ChainAgentVFL backend

FastAPI service for versioned files, CSV ML training (including VFL), batch predictions, FAISS + SentenceTransformers RAG, run/monitoring traces, and optional OpenAI-backed agentic actions. Artifacts live under **`storage/`** in this directory unless `STORAGE_ROOT` overrides it.

## Prerequisites

- **Python** 3.11+ (3.12–3.13 recommended; some ML stacks warn on 3.14+).
- **MySQL** 8.x (or compatible). The default config targets MySQL via PyMySQL.
- Disk/RAM for **sentence-transformers** / **PyTorch** on first use (models download on demand).

## Setup

1. **Database** — default DB name **`agentic-vfl`** (hyphen; quote with backticks in SQL):

   ```bash
   mysql -u root -p < db/init_mysql.sql
   ```

   Or run the same statement in a SQL client (quote the database name with backticks because of the hyphen).

2. **Environment** — from `backend/`:

   - Windows: `copy .env.example .env`
   - Linux/macOS: `cp .env.example .env`

   Edit `DATABASE_URL` if needed. Leave **`STORAGE_ROOT`** empty to use **`backend/storage/`**.

3. **Virtualenv and dependencies**:

   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # Linux/macOS
   pip install -r requirements.txt
   ```

4. **Tables** — optional before first run; the app also runs SQLAlchemy **`create_all`** on startup:

   ```bash
   python scripts/init_orm_tables.py
   ```

## Run the API

Always run from **`backend/`** so `app` imports resolve:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Multiple workers (recommended for responsiveness)

If you run heavy simulation/prediction steps, a **single** Uvicorn worker can get saturated and the dashboard will appear to hang. You can start Uvicorn with multiple worker processes:

```bash
# Windows (PowerShell / cmd)
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Notes:
- **`--reload` + `--workers`**: typically avoid combining in dev; prefer `--reload` while iterating, and `--workers N` when you want responsiveness under load.
- **Worker count**: start with **4** (or roughly `2 * CPU_cores`) and adjust based on CPU/RAM.

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000/docs | Swagger UI |
| http://127.0.0.1:8000/openapi.json | OpenAPI schema |
| http://127.0.0.1:8000/health | Liveness |

### Startup behavior

- Ensures **`storage/`** subfolders exist (`uploads`, `knowledge`, `models`, `predictions`, `reports`, `vector_db`, `training_datasets`).
- Runs **`init_db()`** (`create_all` + small MySQL compatibility checks). For production, prefer **Alembic** (or similar) instead of relying on `create_all` alone.
- Older MySQL databases may get a one-time **`ALTER TABLE`** for `prediction_jobs.results_json` if the column is missing.

## API surface (routers)

| Area | Typical endpoints (prefix `/api/v1` where applicable) |
|------|-----------------------------------------------------|
| Health | `GET /health` |
| Files & datasets | Managed uploads, training/prediction CSVs |
| Training & models | Start training, list jobs, register/delete model versions |
| Predictions | Batch jobs, results, pending purge |
| Agent | Decide, reports, agentic jobs |
| Knowledge base | Upload, query, RAG helpers |
| Simulate | `POST .../simulate/network-row`, `network-row/simple`, `network-traffic`, `network-event` |
| Runs | List runs and events for monitoring |

**Useful reads:** `GET /predictions`, `GET /agent/reports`, `GET /kb/files`, `GET /runs`, `GET /runs/{id}/events`.

## Simulate network traffic (curl)

End-to-end demo (recommended): from **`backend/`** with the server running:

```bash
python scripts/simulate_network_event_demo.py --base http://127.0.0.1:8000
```

**Windows:** use **`curl.exe`** (not PowerShell’s `curl` alias). Avoid bash-style `\` line continuations — use a **single line** or `--data-binary @body.json`.

**Idempotency:** send a fresh **`Idempotency-Key`** per new run; repeating a key can return the **same** run (including a previous **failed** result).

**Minimal example** (`network-event` — tiny schema; real VFL rows need columns aligned to your model):

```bash
curl.exe -sS -X POST "http://127.0.0.1:8000/api/v1/simulate/network-event" -H "Content-Type: application/json" -H "Idempotency-Key: demo-event-001" -d "{\"model_version_public_id\":null,\"columns_csv\":\"bidirectional_duration_ms,bidirectional_packets,label\",\"row_csv\":\"100,5,BENIGN\",\"metadata\":{},\"simulate\":{\"latency_ms\":0}}"
```

**Fixed feature row** (no header; length must match server `VFL_FIXED_COLUMNS` in `app/routers/simulate.py`): `POST /api/v1/simulate/network-row/simple` with JSON `{"values_csv":"<comma-separated-numbers>","metadata":{},"simulate":{}}` — copy a full `values_csv` from `scripts/simulate_network_event_demo.py` (row without trailing `label`).

**After `202` responses**, poll:

```bash
curl.exe -sS "http://127.0.0.1:8000/api/v1/runs/<RUN_ID>"
curl.exe -sS "http://127.0.0.1:8000/api/v1/runs/<RUN_ID>/events"
```

## Optional: HTTP smoke test

```bash
cd backend
python scripts/api_client_demo.py --base http://127.0.0.1:8000
```

Uses **`scripts/data/sample_train.csv`** by default.

## Standalone CLI scripts

Run from **`backend/`**. Notebook-derived helpers live under **`app/notebook_runtime/`**; launchers may `chdir` to the repo root for paths like **`RAG_docs/`**, **`datasets/`**.

| Script | Role |
|--------|------|
| `scripts/api_client_demo.py` | API smoke test |
| `scripts/simulate_network_event_demo.py` | Simulate network traffic + poll run/events |
| `scripts/init_orm_tables.py` | Create tables early |
| `scripts/rag_part1_build_vector_store.py` | RAG index build |
| `scripts/rag_part2_agent_actions.py` | RAG + action plans |
| `scripts/vfl_shap_multiclass.py` | VFL SHAP (multiclass) |
| `scripts/vfl_shap_prediction.py` | VFL SHAP prediction |
| `scripts/scoring_evaluation.py` | Scoring evaluation |
| `scripts/generate_sample_csv.py` | Sample CSV |
| `scripts/build_local_faiss_demo.py` | Local FAISS demo |
| `scripts/merge_notebook_to_task.py` | Regenerate merged runners from `.ipynb` |

Optional **`notebooks/*.ipynb`** — install Jupyter separately if needed.

## Configuration

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | SQLAlchemy URL (default MySQL: `mysql+pymysql://root:test@127.0.0.1:3306/agentic-vfl`) |
| `STORAGE_ROOT` | Artifact root; empty → `backend/storage` |
| `DEBUG` | Verbose logging when enabled (via settings) |
| `OPENAI_API_KEY` | If unset, agent and RAG-LLM paths use mock responses |
| `OPENAI_MODEL` | Chat model id when using OpenAI |
| `EMBEDDING_MODEL` | SentenceTransformers model id for embeddings/RAG |
| `TRUST_CHAIN_ENABLED` | If true, anchor a hash-only trust commitment on local chain after each agentic report is saved |
| `TRUST_CHAIN_RPC_URL` | JSON-RPC URL (default `http://127.0.0.1:8545`) |
| `TRUST_CHAIN_PRIVATE_KEY` | Deployer/signer private key (Hardhat dev key) |
| `TRUST_CHAIN_CONTRACT_ADDRESS` | Deployed `AgenticTrustRegistry` address |
| `TRUST_CHAIN_CHAIN_ID` | Chain id (default `31337`) |
| `TRUST_CHAIN_PAYLOAD_VERSION` | Hash payload version string (default `v1`) |

See **`app/core/config.py`** for defaults such as `rag_chunk_size`, `rag_top_k`, and training split settings.
