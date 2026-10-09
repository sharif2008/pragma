# E2E detect → RAG → reason → chain — latency

Input: `D:\Projects\ChainAgentVFL\experiments\data\eval-100\flows.csv`. N = **100**. System: `RAG_RANKING`. LLM calls = **100**.
Happy path only. Plans stored as generated. No inject. No BERTScore.

Class histogram (true_label): `{"BENIGN": 12, "BOT": 11, "DDOS": 11, "DOS": 11, "FTPPATATOR": 11, "OTHERS": 11, "PORTSCAN": 11, "SSHPATATOR": 11, "WEBATTACK": 11}`.

## Latency (total ms)

Cell = **total ms** over that row’s n flows. **E2E** is the row sum of Detect…Apply. **Total** is the column sum of the nine attack types (n and every step).

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | 12 | 2667.6 | 5338.6 | 33707.5 | 130177.2 | 780.6 | 335.2 | 2131.5 | 175138.2 |
| BOT | 11 | 2445.3 | 3165.6 | 30167.3 | 168031.6 | 704.8 | 317.0 | 2979.5 | 207811.1 |
| DDOS | 11 | 2445.3 | 3271.3 | 29571.9 | 131254.4 | 759.5 | 318.1 | 3581.0 | 171201.5 |
| DOS | 11 | 2445.3 | 3122.0 | 30348.6 | 135873.8 | 790.5 | 325.2 | 4195.2 | 177100.6 |
| FTPPATATOR | 11 | 2445.3 | 3543.8 | 37914.2 | 178283.6 | 649.9 | 266.9 | 2580.5 | 207740.6 |
| OTHERS | 11 | 2445.3 | 3556.1 | 29596.2 | 148809.1 | 736.0 | 305.0 | 2687.8 | 188135.5 |
| PORTSCAN | 11 | 2445.3 | 3480.9 | 33697.8 | 163788.5 | 827.0 | 329.8 | 4693.3 | 209262.6 |
| SSHPATATOR | 11 | 2445.3 | 3152.4 | 29724.1 | 193893.8 | 744.1 | 290.0 | 3589.1 | 233838.8 |
| WEBATTACK | 11 | 2445.3 | 3358.8 | 29142.2 | 170725.7 | 710.1 | 309.3 | 3217.6 | 209909.0 |
| Total | 100 | 22230.0 | 31989.5 | 283869.8 | 1420837.7 | 6702.5 | 2796.5 | 29655.5 | 1780137.9 |

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
| BENIGN | 12 | 222.30 | 444.88 | 2808.96 | 10848.10 | 65.05 | 27.93 | 177.62 | 14594.85 |
| BOT | 11 | 222.30 | 287.78 | 2742.48 | 15275.60 | 64.07 | 28.82 | 270.86 | 18891.92 |
| DDOS | 11 | 222.30 | 297.39 | 2688.35 | 11932.22 | 69.05 | 28.92 | 325.55 | 15563.77 |
| DOS | 11 | 222.30 | 283.82 | 2758.96 | 12352.16 | 71.86 | 29.56 | 381.38 | 16100.05 |
| FTPPATATOR | 11 | 222.30 | 322.16 | 3446.75 | 16207.60 | 59.08 | 26.69 | 258.05 | 20774.06 |
| OTHERS | 11 | 222.30 | 323.28 | 2690.56 | 13528.10 | 66.91 | 27.73 | 244.35 | 17103.23 |
| PORTSCAN | 11 | 222.30 | 316.45 | 3063.44 | 14889.86 | 75.18 | 29.98 | 426.66 | 19023.87 |
| SSHPATATOR | 11 | 222.30 | 286.58 | 2702.19 | 17626.71 | 67.65 | 26.36 | 326.28 | 21258.07 |
| WEBATTACK | 11 | 222.30 | 305.35 | 2649.29 | 15520.52 | 64.55 | 28.12 | 292.51 | 19082.64 |
| Total | 100 | 222.30 | 319.89 | 2838.70 | 14208.38 | 67.03 | 28.25 | 299.55 | 17981.19 |

## Latency (min ms)

Cell = **min ms** among that row’s flows. **Total** is the **sum** of Detect / Retrieve / Rank / … over all attack types (same as the total-ms table), not the global min.

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | 12 | 222.3 | 276.8 | 2394.9 | 8769.7 | 55.1 | 23.9 | 154.9 | 12319.3 |
| BOT | 11 | 222.3 | 265.9 | 2314.5 | 11075.1 | 56.6 | 23.4 | 158.3 | 14431.6 |
| DDOS | 11 | 222.3 | 268.8 | 2410.8 | 8140.5 | 58.9 | 23.8 | 253.5 | 11676.9 |
| DOS | 11 | 222.3 | 266.8 | 2453.0 | 10491.3 | 58.1 | 23.9 | 238.9 | 13960.5 |
| FTPPATATOR | 11 | 222.3 | 269.7 | 2952.1 | 13203.1 | 0.0 | 23.0 | 174.0 | 17216.8 |
| OTHERS | 11 | 222.3 | 261.2 | 2385.2 | 9860.3 | 57.3 | 23.6 | 158.6 | 13349.7 |
| PORTSCAN | 11 | 222.3 | 274.1 | 2592.5 | 11705.2 | 61.2 | 23.0 | 311.6 | 16019.6 |
| SSHPATATOR | 11 | 222.3 | 265.3 | 2470.5 | 14049.3 | 54.4 | 22.6 | 167.8 | 17337.4 |
| WEBATTACK | 11 | 222.3 | 260.6 | 2386.6 | 12151.4 | 56.5 | 22.3 | 152.5 | 15474.0 |
| Total | 100 | 22230.0 | 31989.5 | 283869.8 | 1420837.7 | 6702.5 | 2796.5 | 29655.5 | 1780137.9 |

## Latency (max ms)

Cell = **max ms** among that row’s flows. **Total** is the **sum** of Detect / Retrieve / Rank / … over all attack types (same as the total-ms table), not the global max.

| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|
| BENIGN | 12 | 222.3 | 1411.5 | 3832.7 | 13991.4 | 89.0 | 38.2 | 249.1 | 18489.9 |
| BOT | 11 | 222.3 | 368.1 | 3802.3 | 20412.9 | 83.5 | 53.7 | 830.6 | 23944.7 |
| DDOS | 11 | 222.3 | 357.0 | 3082.1 | 16741.1 | 81.8 | 40.3 | 449.3 | 20338.5 |
| DOS | 11 | 222.3 | 314.0 | 4154.9 | 14957.8 | 96.2 | 45.1 | 480.0 | 18541.2 |
| FTPPATATOR | 11 | 222.3 | 465.0 | 6039.3 | 21509.1 | 98.3 | 32.3 | 397.0 | 25655.9 |
| OTHERS | 11 | 222.3 | 479.7 | 3226.5 | 17848.9 | 85.3 | 30.8 | 379.7 | 22114.1 |
| PORTSCAN | 11 | 222.3 | 420.4 | 3968.9 | 18189.4 | 106.7 | 37.5 | 565.9 | 22032.1 |
| SSHPATATOR | 11 | 222.3 | 333.9 | 3219.0 | 24068.0 | 114.9 | 32.5 | 534.6 | 28069.4 |
| WEBATTACK | 11 | 222.3 | 448.1 | 4058.9 | 20759.6 | 69.8 | 56.7 | 420.8 | 24268.8 |
| Total | 100 | 22230.0 | 31989.5 | 283869.8 | 1420837.7 | 6702.5 | 2796.5 | 29655.5 | 1780137.9 |

## E2E per flow (ms)

Per-flow E2E = Detect + Retrieve + Rank + LLM + Commit + Verify + Apply. **Total** is all n flows (mean / min / max over every flow).

| Attack | n | mean | min | max |
|--------|--:|-----:|----:|----:|
| BENIGN | 12 | 14594.85 | 12319.3 | 18489.9 |
| BOT | 11 | 18891.92 | 14431.6 | 23944.7 |
| DDOS | 11 | 15563.77 | 11676.9 | 20338.5 |
| DOS | 11 | 16100.05 | 13960.5 | 18541.2 |
| FTPPATATOR | 11 | 20774.06 | 17216.8 | 25655.9 |
| OTHERS | 11 | 17103.23 | 13349.7 | 22114.1 |
| PORTSCAN | 11 | 19023.87 | 16019.6 | 22032.1 |
| SSHPATATOR | 11 | 21258.07 | 17337.4 | 28069.4 |
| WEBATTACK | 11 | 19082.64 | 15474.0 | 24268.8 |
| Total | 100 | 17981.19 | 11676.9 | 28069.4 |

### Total by step (per-flow ms)

| Step | n | mean | min | max | sum |
|------|--:|-----:|----:|----:|----:|
| Detect | 100 | 222.30 | 222.3 | 222.3 | 22230.0 |
| Retrieve | 100 | 319.89 | 260.6 | 1411.5 | 31989.5 |
| Rank | 100 | 2838.70 | 2314.5 | 6039.3 | 283869.8 |
| LLM | 100 | 14208.38 | 8140.5 | 24068.0 | 1420837.7 |
| Commit | 100 | 67.03 | 0.0 | 114.9 | 6702.5 |
| Verify | 99 | 28.25 | 22.3 | 56.7 | 2796.5 |
| Apply | 99 | 299.55 | 152.5 | 830.6 | 29655.5 |
| E2E | 99 | 17981.19 | 11676.9 | 28069.4 | 1780137.9 |

## Honest store

| Attack | n | Stored | Applied | Fail | unresolved |
|--------|--:|-------:|--------:|-----:|-----------:|
| BENIGN | 12 | 12 | 24 | 0 | 0 |
| BOT | 11 | 11 | 28 | 0 | 0 |
| DDOS | 11 | 11 | 38 | 0 | 0 |
| DOS | 11 | 11 | 49 | 0 | 0 |
| FTPPATATOR | 11 | 10 | 28 | 1 | 0 |
| OTHERS | 11 | 11 | 31 | 0 | 0 |
| PORTSCAN | 11 | 11 | 54 | 0 | 0 |
| SSHPATATOR | 11 | 11 | 42 | 0 | 0 |
| WEBATTACK | 11 | 11 | 38 | 0 | 0 |
| Total | 100 | 99 | 332 | 1 | 0 |

Total n / Stored / Applied / Fail / unresolved are the sums of the attack-type rows. Applied is `markApplied` action units, not plans.

## Chain fails

- `87486` action_not_whitelisted
