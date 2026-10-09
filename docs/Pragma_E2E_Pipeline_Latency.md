# End-to-End Pipeline Latency of Detection-to-Action Agentic Network Defense

**PRAGMA evaluation note.** This write-up reports the happy-path Detect → Retrieve → Rank → LLM → Commit → Verify → Apply clocks on the frozen `eval-100` fixture. It is intended as a drop-in experimental section for the IEEE paper (complements the blockchain-only Table in `Pragma_v2.tex`, which omitted Detect, RAG, and LLM). Cells in Table I are **total wall time** (sum over that row’s \(n\) flows), not per-flow means.

---

## Abstract

Autonomous network defense that couples vertical federated detection, retrieval-augmented mitigation planning, and blockchain-governed execution is often assumed to be limited by the ledger. We measure the opposite. On 100 CICIDS-2017-style flows processed sequentially through a single ranked-RAG system, total end-to-end wall time is 1780.1 s. Language-model plan generation accounts for 1420.8 s (79.8%), Maximum Marginal Relevance ranking for 283.9 s (15.9%), and the three on-chain steps together for 39.2 s (2.2%). Per-flow mean time is 18.0 s, of which 14.2 s is the LLM. Commit, verify, and apply remain tens to hundreds of milliseconds. The result is that a local smart-contract gate is not the operational bottleneck: decode time of a reasoning-class planner, followed by dense re-ranking, dominates detection-to-action latency.

## Keywords

End-to-end latency, agentic AI, retrieval-augmented generation, vertical federated learning, blockchain enforcement, CICIDS-2017

---

## I. Introduction

Prior PRAGMA evaluation reported blockchain overhead in isolation: on 500 online flows, `anchor` plus `getCommitment` averaged 78.91 ms per flow, with Detect, retrieval, and LLM time omitted. That measurement answers whether a local registry is cheap. It does not answer how long a flow actually spends in the deployed Detect–Reason–Commit–Apply path.

This note reports that missing clock. One system (`RAG_RANKING`) is run on a frozen 100-flow fixture spanning nine CICIDS-style labels. Plans are stored as generated: no injection, no mutation, no BERTScore. The headline quantity is **total milliseconds per attack type and pipeline step**, with end-to-end (E2E) defined as the row sum of the seven stages and Total defined as the column sum of the nine labels.

The measurement supports three claims. First, language-model generation is the dominant cost, not detection and not the ledger. Second, ranking (MMR re-embedding) is the only other first-order term. Third, attack class changes LLM time by nearly \(2\times\) while leaving Detect, retrieve, and chain clocks nearly constant.

---

## II. Experimental Setup

**Fixture.** Input is `experiments/data/eval-100/flows.csv` (\(N=100\)). The class histogram by `true_label` is BENIGN 12 and 11 flows each of BOT, DDOS, DOS, FTPPATATOR, OTHERS, PORTSCAN, SSHPATATOR, and WEBATTACK. Latency rows are grouped by true label; the detector label \(\hat{y}\) is what Commit stores.

**System.** One planner, one LLM call per flow (100 calls). Retrieval is hybrid: FAISS \(N=80\) and BM25 \(N=80\), fused by reciprocal rank fusion \(1/(60+\mathrm{rank})\), then MMR with \(\lambda=0.5\) keeping 20 children and expanding to 5 parent sections. The prompt is `create_agentic_orchestration_prompt` with knowledge base and agentic conditions enabled. \(W[\hat{y}]\) is taken from `attack_options.json`. The model is `OPENAI_MODEL` (default `gpt-6-luna`) via Chat Completions.

**Pipeline clocks.**

| Stage | What is timed |
|-------|----------------|
| Detect | Three-party VFL + KernelSHAP, amortized per flow (chunk wall time / \(n\)) |
| Retrieve | FAISS + BM25 + RRF |
| Rank | MMR |
| LLM | Mitigation Plan JSON |
| Commit | `storePlan` |
| Verify | `getPlan` / `isInPlan` |
| Apply | honest `markApplied` over the plan’s units |
| E2E | sum of the seven stages |

**Chain.** Hardhat at `127.0.0.1:8545`, chain ID 31337, `AgenticTrustRegistry`. `planId` is `{split_index}-RAG_RANKING-honest`. On revert the runner records the fail and continues; it does not rewrite the plan. `attack_type_unresolved` must be 0.

**Accounting.** A cell in Table I is the **sum** of that stage over the row’s \(n\) flows, in milliseconds. E2E is the row sum of Detect through Apply. Total is the column sum of the nine attack-type rows. Per-flow means in the text are those sums divided by \(n\) (Total by 100). Detect is identical across labels because SHAP ran in chunks and the chunk cost is amortized.

---

## III. Results

### A. Total wall time (headline)

Table I is the required latency matrix: attack type \(\times\) stage, Total last. Units are milliseconds. SSHPATATOR is the slowest class (233.8 s over 11 flows) because its LLM column is the largest (193.9 s). DDOS is the fastest non-BENIGN class (171.2 s) with the second-lowest LLM total (131.3 s). BENIGN has the lowest LLM total (130.2 s) despite \(n=12\), so its E2E (175.1 s) sits near DDOS rather than at the top of the table.

**Table I.** Total wall-clock time (ms) by attack type and pipeline stage. Each cell is the sum over that row’s \(n\) flows. E2E is the row sum of Detect–Apply. Total is the column sum of the nine labels.

| Attack | \(n\) | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |
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
| **Total** | **100** | **22230.0** | **31989.5** | **283869.8** | **1420837.7** | **6702.5** | **2796.5** | **29655.5** | **1780137.9** |

Over all 100 flows the pipeline spends 1780.1 s. Table II restates the Total row as a share of that E2E sum.

**Table II.** Contribution of each stage to total E2E time (\(N=100\)). Mean is Total / 100 except Verify and Apply, which are over the 99 stored plans.

| Stage | Sum (s) | Share of E2E | Mean (ms / flow) |
|-------|--------:|-------------:|-----------------:|
| LLM | 1420.8 | 79.8% | 14208 |
| Rank | 283.9 | 15.9% | 2839 |
| Retrieve | 32.0 | 1.8% | 320 |
| Apply | 29.7 | 1.7% | 300 |
| Detect | 22.2 | 1.2% | 222 |
| Commit | 6.7 | 0.4% | 67 |
| Verify | 2.8 | 0.2% | 28 |
| Commit + Verify + Apply | 39.2 | 2.2% | — |
| E2E | 1780.1 | 100% | 17981 |

### B. Per-flow means

Mean E2E is 18.0 s per flow (min 11.7 s on a DDOS row, max 28.1 s on an SSHPATATOR row). Mean LLM time is 14.2 s (min 8.1 s, max 24.1 s). Mean Rank is 2.8 s. Mean Retrieve is 0.32 s after the first-flow encoder warmup (the BENIGN retrieve max of 1.41 s is that warmup). Detect is 222.3 ms for every label by construction of the amortized chunk clock.

LLM mean varies by class more than any other stage: BENIGN 10.8 s, DDOS 11.9 s, DOS 12.4 s, OTHERS 13.5 s, PORTSCAN 14.9 s, BOT 15.3 s, WEBATTACK 15.5 s, FTPPATATOR 16.2 s, SSHPATATOR 17.6 s. Rank is comparatively stable (2.65–3.06 s) except FTPPATATOR (3.45 s). Chain means stay in a narrow band: Commit 59–75 ms, Verify 26–30 ms, Apply 178–427 ms (Apply tracks the number of `markApplied` units, not the attack name).

### C. Honest store

Of 100 generated plans, 99 were stored and 332 action units were applied. `attack_type_unresolved` is 0: every flow had a usable \(\hat{y}\) for the registry. One FTPPATATOR plan (`split_index` 87486) reverted at `storePlan` with `action_not_whitelisted` (the planner named a control absent from \(W[\mathrm{FTPPATATOR}]\)); Commit is recorded as 0.0 ms on that row and Verify/Apply are skipped. The gate did what RQ5 requires: an off-catalog token never reached Apply. It is a planner catalog error, not a latency artifact.

### D. Stage ablation of end-to-end latency

This experiment uses a single system (`RAG_RANKING`), so the ablation is **compositional**: each pipeline stage is removed from the measured E2E sum while the other clocks stay as recorded. It is not a re-run of `LLM_only` or `RAG_No_Ranking`. The baseline is the Total E2E of Table I, \(1780.1\)\,s. Remaining time after removing stage \(s\) is \(\mathrm{E2E}-\mathrm{sum}(s)\). Reduction is \(\mathrm{sum}(s)/\mathrm{E2E}\). Verify, Apply, and E2E are over the \(99\) stored plans; Detect, Retrieve, Rank, LLM, and Commit include the one reverted FTPPATATOR flow.

**Table III.** Leave-one-stage-out ablation of E2E wall time (\(N=100\) fixture, \(99\) complete traces). Remaining and mean use the Table I Total E2E of \(1780.1\)\,s as the full-pipeline baseline.

| Ablated stage | Remaining total (s) | Remaining mean (s/flow) | Reduction |
|---------------|--------------------:|------------------------:|----------:|
| None (full pipeline) | 1780.1 | 17.98 | — |
| LLM | 359.3 | 3.59 | 79.8% |
| Rank (MMR) | 1496.3 | 14.96 | 15.9% |
| Retrieve (FAISS+BM25+RRF) | 1748.1 | 17.48 | 1.8% |
| Apply (`markApplied`) | 1750.5 | 17.50 | 1.7% |
| Detect (VFL+SHAP) | 1757.9 | 17.58 | 1.2% |
| Commit (`storePlan`) | 1773.4 | 17.73 | 0.4% |
| Verify (`getPlan`) | 1777.3 | 17.77 | 0.2% |
| Chain (Commit+Verify+Apply) | 1741.0 | 17.41 | 2.2% |
| Reason (Retrieve+Rank+LLM) | 43.4 | 0.43 | 97.6% |

Removing the language-model call is the only ablation that changes the operating regime (from \(18\)\,s/flow to \(3.6\)\,s/flow). Removing MMR ranking is the only other first-order cut (\(2.8\)\,s/flow). Removing the entire chain changes E2E by \(0.39\)\,s/flow. Removing Detect changes it by \(0.22\)\,s/flow.

**Table IV.** Layer aggregation of the same clocks. Reason is Retrieve+Rank+LLM. Chain is Commit+Verify+Apply. Share is against Total E2E \(1780.1\)\,s.

| Layer | Stages | Sum (s) | Share of E2E | Mean (ms/flow) |
|-------|--------|--------:|-------------:|---------------:|
| Detect | VFL + KernelSHAP | 22.2 | 1.2% | 222 |
| Reason | Retrieve + Rank + LLM | 1736.7 | 97.6% | 17367 |
| Chain | Commit + Verify + Apply | 39.2 | 2.2% | 392 |
| E2E | seven stages | 1780.1 | 100% | 17981 |

**Table V.** Per-class mean E2E and the LLM share of that class mean. Detect is \(222.3\)\,ms for every label. Chain means stay in \(265\)--\(532\)\,ms.

| Attack | Mean E2E (s) | Mean LLM (s) | LLM share of class E2E | Mean Rank (s) |
|--------|-------------:|-------------:|-----------------------:|--------------:|
| BENIGN | 14.59 | 10.85 | 74.3% | 2.81 |
| DDOS | 15.56 | 11.93 | 76.7% | 2.69 |
| DOS | 16.10 | 12.35 | 76.7% | 2.76 |
| OTHERS | 17.10 | 13.53 | 79.1% | 2.69 |
| BOT | 18.89 | 15.28 | 80.9% | 2.74 |
| PORTSCAN | 19.02 | 14.89 | 78.3% | 3.06 |
| WEBATTACK | 19.08 | 15.52 | 81.3% | 2.65 |
| FTPPATATOR | 20.77 | 16.21 | 78.0% | 3.45 |
| SSHPATATOR | 21.26 | 17.63 | 82.9% | 2.70 |
| All flows | 17.98 | 14.21 | 79.8% | 2.84 |

Class-conditional E2E spans \(14.6\)--\(21.3\)\,s (\(1.46\times\)). LLM time spans \(10.8\)--\(17.6\)\,s (\(1.62\times\)). Rank is stable except FTPPATATOR (\(3.45\)\,s). The class ranking of E2E is the class ranking of LLM time; Detect and the chain do not reorder it.

---

## IV. Analysis

**The ledger is not the bottleneck.** Combined Commit + Verify + Apply is 2.2% of E2E (Table III). That is consistent with the earlier 500-flow blockchain-only table (~79 ms mean store+verify): a local Hardhat registry adds tens of milliseconds. Reporting only those milliseconds understates operational latency by more than an order of magnitude. Ablating the entire chain leaves 17.41 s/flow.

**The LLM is the bottleneck.** 79.8% of wall time is Mitigation Plan generation. Removing it is the only ablation that changes the operating regime (18.0 → 3.59 s/flow). The prompt is large (~9k tokens: five parent sections, SHAP JSON, \(W[\hat{y}]\), agentic cards, and a strict JSON schema that demands per-action reasoning). Chat Completions on a reasoning-class model (`gpt-6-luna`) spends billed completion tokens on hidden reasoning as well as the visible plan. Class-conditional LLM time (BENIGN 10.8 s vs SSHPATATOR 17.6 s) tracks how much the model writes, not retrieve or Detect.

**Rank is the second term.** MMR re-embeds the RRF candidate pool. At 15.9% of E2E (~2.8 s/flow) it is larger than Detect, Retrieve, and the entire chain combined. Ablating Rank leaves 14.96 s/flow. Hybrid retrieve itself is cheap (1.8%) once the sentence encoder is warm.

**Detect is cheap and flat.** Amortized VFL+SHAP is 1.2% of E2E and identical across labels. Ablating Detect leaves 17.58 s/flow. Any claim that federated inference dominates agentic response time is not supported on this fixture.

**Class effects are LLM effects.** E2E rank order of the nine labels is essentially the LLM column order (Table V). Retrieve, Rank (except FTPPATATOR), and chain columns do not reorder the table. Reason as a layer (Retrieve+Rank+LLM) is 97.6% of E2E; Detect+Chain together are 3.4%.

---

## V. Implications

1. **Enforcement cost is not the deployment objection.** A local (and, by extrapolation, a permissioned) registry can sit on the critical path without setting the SLA. The SLA is set by plan generation.
2. **Latency engineering should start at decode.** Shorter JSON, lower reasoning effort, prompt-cache reuse of the static instruction and catalog, and fewer RAG parents reduce the 14 s term; they do not change the 67 ms `storePlan`.
3. **Ranking is the only other lever that moves E2E by seconds.** Skipping MMR, caching child embeddings, or shrinking the MMR pool would cut the 2.8 s term. Retrieval fusion (FAISS+BM25+RRF) is already sub-second.
4. **Happy-path success is not the same as catalog compliance.** 99/100 stores with unresolved \(=0\) shows the path is live. The single whitelist revert shows Reason can still emit an illegal token; that failure belongs to the planner, and the contract correctly refused it.

---

## VI. Limitations

\(N=100\) is a stratified slice, not the 1000-flow default fixture. Detect time is chunk-amortized, so the Detect column cannot be read as per-flow SHAP. The blockchain is one Hardhat node, not a WAN consortium. Apply writes receipts, not firewall or SOAR actuations. The LLM is Chat Completions without an explicit reasoning-effort cap; a Responses API run with `effort=low` is a separate smoke measurement and is not mixed into Table I. No BERTScore or inject-block study is included: this experiment clocks the unmodified success path.

---

## VII. Conclusion

On 100 sequential CICIDS-style flows, PRAGMA’s detection-to-action pipeline spends 1780.1 s in total, 18.0 s per flow. Language-model mitigation planning is 79.8% of that time, MMR ranking 15.9%, and blockchain commit/verify/apply 2.2%. Per-class differences in E2E are almost entirely differences in LLM time. The earlier finding that on-chain commitment costs tens of milliseconds remains true; it is not the quantity that governs end-to-end response delay. Practical acceleration of agentic network defense in this architecture is a planner and ranker problem, not a ledger problem.

---

## Appendix: correspondence with `report.md`

Source: `experiments/e2e-detect-chain/report.md`, section “Latency (total ms)”. Mean / min / max matrices in that file use the same clocks; Total in the min and max matrices is the same column-sum as Table I, not a global min or max. Honest-store Total: \(n=100\), stored 99, applied 332 units, fail 1, unresolved 0.
