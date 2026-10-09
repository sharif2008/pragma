# E2E detect → RAG → reason → chain — latency

Input: `D:\Projects\ChainAgentVFL\experiments\data\eval-10\flows.csv`. N = **10**. System: `RAG_RANKING`. LLM calls = **10**.
Happy path only. Plans stored as generated. No inject. No BERTScore.

Class histogram (true_label): `{"BENIGN": 2, "BOT": 1, "DDOS": 1, "DOS": 1, "FTPPATATOR": 1, "OTHERS": 1, "PORTSCAN": 1, "SSHPATATOR": 1, "WEBATTACK": 1}`.

## Latency (total ms)

Cell = **total ms** over that row’s n flows. **E2E** is the row sum of Detect…Apply. **Total** is the column sum of the nine attack types (n and every step).

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | 2 | 4528.2 | 3071.4 | 12600.9 | 24070.9 | 596.2 | 77.6 | 716.6 | 45661.8 |
| BOT | 1 | 2264.1 | 2618.7 | 4748.8 | 20640.7 | 348.6 | 43.1 | 547.9 | 31211.9 |
| DDOS | 1 | 2264.1 | 461.3 | 2933.0 | 12688.6 | 133.5 | 33.1 | 384.5 | 18898.1 |
| DOS | 1 | 2264.1 | 378.2 | 2821.5 | 15831.4 | 139.2 | 40.5 | 945.2 | 22420.1 |
| FTPPATATOR | 1 | 2264.1 | 347.0 | 4457.8 | 23862.7 | 103.6 | 43.4 | 367.7 | 31446.3 |
| OTHERS | 1 | 2264.1 | 784.0 | 4131.0 | 13307.3 | 94.4 | 31.3 | 254.7 | 20866.8 |
| PORTSCAN | 1 | 2264.1 | 487.0 | 4909.4 | 14806.6 | 104.8 | 48.3 | 586.2 | 23206.4 |
| SSHPATATOR | 1 | 2264.1 | 347.0 | 5523.3 | 16511.7 | 112.0 | 28.9 | 419.3 | 25206.3 |
| WEBATTACK | 1 | 2264.1 | 331.9 | 3122.8 | 25318.8 | 93.2 | 38.3 | 281.1 | 31450.2 |
| Total | 10 | 22641.0 | 8826.5 | 45248.5 | 167038.7 | 1725.5 | 384.5 | 4503.2 | 250367.9 |

| Step | Clock |
|------|--------|
| Detect | VFL + SHAP, amortized per flow |
| Retrieve | FAISS + BM25 + RRF |
| Rank | MMR |
| LLM | Mitigation Plan |
| Commit | `storePlan` |
| Verify | `getPlan` / `isInPlan` |
| Apply | honest `markApplied` (per plan) |
| E2E | sum of the seven steps |

## Latency (mean ms)

Cell = **mean ms** per flow in that row. **Total** is mean over all n flows.

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | 2 | 2264.10 | 1535.70 | 6300.45 | 12035.45 | 298.10 | 38.80 | 358.30 | 22830.90 |
| BOT | 1 | 2264.10 | 2618.70 | 4748.80 | 20640.70 | 348.60 | 43.10 | 547.90 | 31211.90 |
| DDOS | 1 | 2264.10 | 461.30 | 2933.00 | 12688.60 | 133.50 | 33.10 | 384.50 | 18898.10 |
| DOS | 1 | 2264.10 | 378.20 | 2821.50 | 15831.40 | 139.20 | 40.50 | 945.20 | 22420.10 |
| FTPPATATOR | 1 | 2264.10 | 347.00 | 4457.80 | 23862.70 | 103.60 | 43.40 | 367.70 | 31446.30 |
| OTHERS | 1 | 2264.10 | 784.00 | 4131.00 | 13307.30 | 94.40 | 31.30 | 254.70 | 20866.80 |
| PORTSCAN | 1 | 2264.10 | 487.00 | 4909.40 | 14806.60 | 104.80 | 48.30 | 586.20 | 23206.40 |
| SSHPATATOR | 1 | 2264.10 | 347.00 | 5523.30 | 16511.70 | 112.00 | 28.90 | 419.30 | 25206.30 |
| WEBATTACK | 1 | 2264.10 | 331.90 | 3122.80 | 25318.80 | 93.20 | 38.30 | 281.10 | 31450.20 |
| Total | 10 | 2264.10 | 882.65 | 4524.85 | 16703.87 | 172.55 | 38.45 | 450.32 | 25036.79 |

## Latency (min ms)

Cell = **min ms** among that row’s flows. **Total** is the **sum** of Detect / Retrieve / Rank / … over all attack types (same as the total-ms table), not the global min.

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | 2 | 2264.1 | 451.4 | 6185.0 | 9579.4 | 107.9 | 35.5 | 244.5 | 18867.8 |
| BOT | 1 | 2264.1 | 2618.7 | 4748.8 | 20640.7 | 348.6 | 43.1 | 547.9 | 31211.9 |
| DDOS | 1 | 2264.1 | 461.3 | 2933.0 | 12688.6 | 133.5 | 33.1 | 384.5 | 18898.1 |
| DOS | 1 | 2264.1 | 378.2 | 2821.5 | 15831.4 | 139.2 | 40.5 | 945.2 | 22420.1 |
| FTPPATATOR | 1 | 2264.1 | 347.0 | 4457.8 | 23862.7 | 103.6 | 43.4 | 367.7 | 31446.3 |
| OTHERS | 1 | 2264.1 | 784.0 | 4131.0 | 13307.3 | 94.4 | 31.3 | 254.7 | 20866.8 |
| PORTSCAN | 1 | 2264.1 | 487.0 | 4909.4 | 14806.6 | 104.8 | 48.3 | 586.2 | 23206.4 |
| SSHPATATOR | 1 | 2264.1 | 347.0 | 5523.3 | 16511.7 | 112.0 | 28.9 | 419.3 | 25206.3 |
| WEBATTACK | 1 | 2264.1 | 331.9 | 3122.8 | 25318.8 | 93.2 | 38.3 | 281.1 | 31450.2 |
| Total | 10 | 22641.0 | 8826.5 | 45248.5 | 167038.7 | 1725.5 | 384.5 | 4503.2 | 250367.9 |

## Latency (max ms)

Cell = **max ms** among that row’s flows. **Total** is the **sum** of Detect / Retrieve / Rank / … over all attack types (same as the total-ms table), not the global max.

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | 2 | 2264.1 | 2620.0 | 6415.9 | 14491.5 | 488.3 | 42.1 | 472.1 | 26794.0 |
| BOT | 1 | 2264.1 | 2618.7 | 4748.8 | 20640.7 | 348.6 | 43.1 | 547.9 | 31211.9 |
| DDOS | 1 | 2264.1 | 461.3 | 2933.0 | 12688.6 | 133.5 | 33.1 | 384.5 | 18898.1 |
| DOS | 1 | 2264.1 | 378.2 | 2821.5 | 15831.4 | 139.2 | 40.5 | 945.2 | 22420.1 |
| FTPPATATOR | 1 | 2264.1 | 347.0 | 4457.8 | 23862.7 | 103.6 | 43.4 | 367.7 | 31446.3 |
| OTHERS | 1 | 2264.1 | 784.0 | 4131.0 | 13307.3 | 94.4 | 31.3 | 254.7 | 20866.8 |
| PORTSCAN | 1 | 2264.1 | 487.0 | 4909.4 | 14806.6 | 104.8 | 48.3 | 586.2 | 23206.4 |
| SSHPATATOR | 1 | 2264.1 | 347.0 | 5523.3 | 16511.7 | 112.0 | 28.9 | 419.3 | 25206.3 |
| WEBATTACK | 1 | 2264.1 | 331.9 | 3122.8 | 25318.8 | 93.2 | 38.3 | 281.1 | 31450.2 |
| Total | 10 | 22641.0 | 8826.5 | 45248.5 | 167038.7 | 1725.5 | 384.5 | 4503.2 | 250367.9 |

## E2E per flow (ms)

Per-flow E2E = Detect + Retrieve + Rank + LLM + Commit + Verify + Apply. **Total** is all n flows (mean / min / max over every flow).

| Attack | n | mean | min | max |
|--------|--:|-----:|----:|----:|
| BENIGN | 2 | 22830.90 | 18867.8 | 26794.0 |
| BOT | 1 | 31211.90 | 31211.9 | 31211.9 |
| DDOS | 1 | 18898.10 | 18898.1 | 18898.1 |
| DOS | 1 | 22420.10 | 22420.1 | 22420.1 |
| FTPPATATOR | 1 | 31446.30 | 31446.3 | 31446.3 |
| OTHERS | 1 | 20866.80 | 20866.8 | 20866.8 |
| PORTSCAN | 1 | 23206.40 | 23206.4 | 23206.4 |
| SSHPATATOR | 1 | 25206.30 | 25206.3 | 25206.3 |
| WEBATTACK | 1 | 31450.20 | 31450.2 | 31450.2 |
| Total | 10 | 25036.79 | 18867.8 | 31450.2 |

### Total by step (per-flow ms)

| Step | n | mean | min | max | sum |
|------|--:|-----:|----:|----:|----:|
| Detect | 10 | 2264.10 | 2264.1 | 2264.1 | 22641.0 |
| Retrieve | 10 | 882.65 | 331.9 | 2620.0 | 8826.5 |
| Rank | 10 | 4524.85 | 2821.5 | 6415.9 | 45248.5 |
| LLM | 10 | 16703.87 | 9579.4 | 25318.8 | 167038.7 |
| Commit | 10 | 172.55 | 93.2 | 488.3 | 1725.5 |
| Verify | 10 | 38.45 | 28.9 | 48.3 | 384.5 |
| Apply | 10 | 450.32 | 244.5 | 945.2 | 4503.2 |
| E2E | 10 | 25036.79 | 18867.8 | 31450.2 | 250367.9 |

## Honest store

| Attack | n | Stored | Applied | Fail | unresolved |
|--------|--:|-------:|--------:|-----:|-----------:|
| BENIGN | 2 | 2 | 4 | 0 | 0 |
| BOT | 1 | 1 | 4 | 0 | 0 |
| DDOS | 1 | 1 | 3 | 0 | 0 |
| DOS | 1 | 1 | 5 | 0 | 0 |
| FTPPATATOR | 1 | 1 | 3 | 0 | 0 |
| OTHERS | 1 | 1 | 2 | 0 | 0 |
| PORTSCAN | 1 | 1 | 5 | 0 | 0 |
| SSHPATATOR | 1 | 1 | 3 | 0 | 0 |
| WEBATTACK | 1 | 1 | 2 | 0 | 0 |
| Total | 10 | 10 | 31 | 0 | 0 |

Total n / Stored / Applied / Fail / unresolved are the sums of the attack-type rows. Applied is `markApplied` action units, not plans.

## Chain fails

None.
