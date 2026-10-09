# Plan: on-chain attack matrix + authorization stories

Source of truth: [`SPEC.md`](SPEC.md). One task. One report under `experiments/agentic-attack/`.

## Not an e2e experiment

Do **not** run `attack_monitor.py`. Do **not** write `experiments/e2e/`. Do **not** ingest a CSV, call Detect, retrieve RAG, or call the LLM.

Stage **A** already exists: a mitigation-plans JSON passed as `--plans`. Stage **Auth** reads `experiments/agentic-attack/auth_plans.json`. Neither file is regenerated.

## Why one report

The scale matrix (A1–A12, A21–A25) and the authorization stories (A13–A20) share the same contract. They are different attacks. They share one markdown table with IDs `A` plus a sequential number: **Threat** (Table III class) next to **How this works** (the injection that represents that threat). Do not label that second column Simulation.

## Prerequisite (SPEC)

- `--plans PATH` — any mitigation-plans JSON. `--system NAME` is optional.
- `--auth-plans` defaults to `experiments/agentic-attack/auth_plans.json`. `--skip-auth` omits Stage Auth. `--auth-only` runs only B1–B8.
- Hardhat already up (`http://127.0.0.1:8545`, chain ID `31337`). This job does not start or stop it.
- `AgenticTrustRegistry` deployed and seeded; `TRUST_CHAIN_*` plus executor keys in `backend/.env`
- Do not read or write off-chain audit logs
- Do not edit `docs/Pragma_v2.tex`

## Steps

1. **Script** — `backend/scripts/agentic_attack_eval.py --plans PATH`. After the matrix, it calls `run_auth_cases` unless `--skip-auth`.
2. **Stage B / C / L** — honest store, inject C1–C6 and T4–T7, route L1/L2.
3. **Stage Auth** — B1–B8 against `--auth-plans`. Write `auth.jsonl` in the same folder.
4. **Report** — one `report.md`. `--report-only` rebuilds it from jsonl and does not re-run the chain.

```
cd backend
python scripts/agentic_attack_eval.py --plans PATH
python scripts/agentic_attack_eval.py --plans PATH --system RAG_RANKING
python scripts/agentic_attack_eval.py --plans PATH --skip-auth
python scripts/agentic_attack_eval.py --auth-only
python scripts/agentic_attack_eval.py --plans PATH --report-only
```

Leave `hardhat-blockchain` running.

## Done when

```
experiments/agentic-attack/
  auth_plans.json    # input; not overwritten
  auth.jsonl         # A13–A20
  honest.jsonl
  attacks.jsonl
  loops.jsonl
  report.md          # A1–A25
  outcomes_a1_a25.png
  honest_store.png
  auth_a13_a20.png
  latency.png
  table_a1_a25.png
  manifest.json
  latency.json
```

Injected Block.% = 100; tamper accepted = 0; auth 8/8. Input plan files untouched. Paper tex untouched. No `experiments/agentic-auth/` and no `.agent/pragma-agentic-auth/`.
