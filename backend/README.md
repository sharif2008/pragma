# ChainAgentVFL backend

FastAPI service for file versioning, CSV ML training, batch predictions, FAISS RAG, and optional OpenAI agentic decisions. Data and artifacts use **`storage/`** under this directory unless overridden.

## Prerequisites

- **Python** 3.11+ (3.12–3.13 recommended; some ML stacks warn on 3.14+).
- **MySQL** 8.x (or compatible) with a database created for the app.
- Enough disk/RAM for **sentence-transformers** / **PyTorch** on first run (models download on demand).

## Setup

1. **Create the MySQL database** — default name **`agentic-vfl`** (hyphen; backticks in SQL). From **`backend`**:

   ```bash
   mysql -u root -p < db/init_mysql.sql
   ```

   Or:

   ```sql
   CREATE DATABASE `agentic-vfl` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

2. **Create ORM tables** (optional; the API also runs `create_all` on startup):

   ```bash
   cd backend
   python scripts/init_orm_tables.py
   ```

3. **Environment file** — copy the example and edit:

   ```bash
   cd backend
   ```

   Windows: `copy .env.example .env` — Linux/macOS: `cp .env.example .env`

   Default example: user **`root`**, password **`test`**, database **`agentic-vfl`**:

   ```env
   DATABASE_URL=mysql+pymysql://root:test@127.0.0.1:3306/agentic-vfl
   ```

   Leave **`STORAGE_ROOT`** empty to use **`backend/storage/`**.

4. **Install dependencies**:

   ```bash
   cd backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   On Linux/macOS, activate with `source .venv/bin/activate`.

## Run the API

From the **`backend`** directory (so `app` imports resolve):

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

On first startup the app creates **`storage/`** subfolders and applies SQLAlchemy **`create_all`** to your MySQL schema (for production, prefer migrations such as Alembic).

**Tracking / history:** `GET /predictions?limit=100&offset=0` lists inference jobs; `GET /agent/reports` and `GET /agent/reports/{public_id}` list agentic action records. Knowledge base entries: `GET /kb/files`.

## Optional: smoke-test the HTTP API

With the server running:

```bash
cd backend
python scripts/api_client_demo.py --base http://127.0.0.1:8000
```

Uses **`scripts/data/sample_train.csv`** by default.

## Standalone CLI scripts (no `.ipynb` required)

Logic lives in **`app/notebook_runtime/`** (copied/merged from the old notebook utils). Run from **`backend`**; each launcher **`chdir`s to the repo root** so paths like **`RAG_docs/`**, **`datasets/`** match the notebooks.

```bash
cd backend
python scripts/rag_part1_build_vector_store.py   # RAG index build
python scripts/rag_part2_agent_actions.py        # RAG + action plans
python scripts/vfl_shap_multiclass.py
python scripts/vfl_shap_prediction.py
python scripts/scoring_evaluation.py
python scripts/generate_sample_csv.py
python scripts/build_local_faiss_demo.py
python scripts/init_orm_tables.py
python scripts/api_client_demo.py
```

- **Regenerate** merged runners after editing an ipynb: `python scripts/merge_notebook_to_task.py`
- Optional **`backend/notebooks/*.ipynb`** remain for interactive use; install Jupyter separately if needed.

## Configuration summary

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | MySQL URL, e.g. `mysql+pymysql://root:test@127.0.0.1:3306/agentic-vfl` |
| `STORAGE_ROOT` | Override artifact root; default `backend/storage` |
| `OPENAI_API_KEY` | If unset, agent and RAG-LLM use mock responses |
| `OPENAI_MODEL` | Chat model id when using OpenAI |
| `EMBEDDING_MODEL` | SentenceTransformers model id for RAG |
