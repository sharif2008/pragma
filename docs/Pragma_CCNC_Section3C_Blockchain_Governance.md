# III-C. Blockchain Governance Layer (Commit)

Section III-B forwards a single object \(\mathrm{plan}_i\): the Mitigation Plans JSON whose `action` names were copied from \(W[\hat{y}_i]\) and whose `network_tier` is exactly Access / ISP, Perimeter / IDS, or Endpoint / EDR. That object is still a **proposal**. Reasoning is probabilistic. A language model can emit a fluent action that is still harmful on a live path; a poisoned or substituted plan can change between generation and dispatch; a policy paragraph in a prompt can be ignored. Autonomous defense therefore needs a pre-execution governance layer that cannot hallucinate mitigations and cannot be talked into accepting them.

A smart contract is a program that lives on a ledger and executes exactly the rules written in its code. Callers submit transactions; the network orders them; every honest replica applies the same function to the same state. We use that property for two jobs only: an **unpromptable per-attack whitelist**, and a **write-once digest** of the plan produced for this detection cycle. The contract does not classify traffic, does not retrieve policy, and does not dispatch. Those remain Detect, Reason, and Apply.

\[
\underbrace{\mathrm{plan}_i}_{\text{III-B}}
\;\xrightarrow{\text{canonicalize}}\;
P
\;\xrightarrow{\mathrm{SHA256}}\;
c
\;\xrightarrow{\mathrm{anchor}}\;
\text{ledger}[k_{\mathrm{agent}},k_{\mathrm{report}}]
\;\xrightarrow{\mathrm{TrustAnchored}}\;
\text{SOC / Apply}.
\]

---

**What is stored off-chain versus on-chain.** After Reason returns, the Mitigation Plans object is persisted off-chain (identity, structured plan, retrieval context). The backend then builds a canonical payload \(P\) with fixed keys, sorted lexicographically, compact separators, and UTC timestamps so two encodings of the same report cannot produce two hashes:

- payload version;
- report and prediction identities;
- optional row index (the same Detection row Reason used);
- creation time;
- structured \(\mathrm{plan}_i\) (threat level, primary/supporting units, tiers, reasoning);
- retrieval context used in the prompt.

\[
c = \mathrm{SHA256}(\mathrm{canonicalJSON}(P)).
\]

Telemetry, the 192-d fusion vector, SHAP tables, and free-form model prose are **not** required on the ledger. Only the 32-byte digest \(c\) is written. Confidentiality is the point: an auditor who holds \(P\) can re-hash and compare; an observer of the chain cannot reconstruct the flow or the prompt.

---

**Addressing and keys.** The registry never stores human-readable identifiers. Four `bytes32` keys are derived before the transaction:

| Key | How it is formed | Why |
|-----|------------------|-----|
| \(k_{\mathrm{agent}}\) | SHA-256 of the agent-job identity | Binds the cycle to the party that requested the plan |
| \(k_{\mathrm{report}}\) | SHA-256 of the report identity | One commitment slot per plan |
| \(k_a\) | \(\mathrm{keccak256}(\hat{y}_i)\) | Attack class as the contract sees it (same UTF-8 label Detect/Reason use) |
| \(k_u\) | \(\mathrm{keccak256}(u)\) | Action string as an exact catalog token (`limit rate`, not “rate-limit”) |

Lookups are \(W[k_a][k_u]\) and \(\mathrm{commitments}[k_{\mathrm{agent}}][k_{\mathrm{report}}]\). A paraphrase that Reason was forbidden to emit also fails here: `limit rate` and `rate limiting` are different keys.

---

**Job 1 — per-attack whitelist \(W\).** At deploy, the owner seeds \(W[a,u]\) from the same catalog Reason was allowed to name (CICIDS-2017 grouping: BENIGN, DDOS, DOS, SSHPATATOR, FTPPATATOR, PORTSCAN, WEBATTACK, BOT, OTHERS). After seed, ordinary callers cannot add a name. `isActionWhitelisted(k_a, k_u)` is a view; it does not depend on the prompt. An action that is legal for DDoS and illegal for the predicted PortScan fails even if the sentence is fluent. That is the failure a policy paragraph cannot close: the model cannot talk a new token onto \(W\).

---

**Job 2 — write-once anchor.** `anchor(k_agent, k_report, c)` requires every argument nonzero and \(\mathrm{commitments}[k_{\mathrm{agent}}][k_{\mathrm{report}}]=0\). A second write is rejected (`already_anchored`). The contract emits `TrustAnchored`. Later readers call `getCommitment` and compare to a fresh SHA-256 of the stored \(P\). Three outcomes are distinguishable:

1. hashes match — \(P\) is the plan that was committed;
2. hashes differ — \(P\) was edited after Commit (silent substitution);
3. commitment is empty — this cycle was never anchored, so Apply must not proceed.

Immutability here is not “the blockchain stores the network.” It is three enforceable facts: the allowed action set for each attack is on-chain and not promptable; the plan for this cycle has a single committed digest; any later apply must bind to both.

---

**What the contract will not check (and why Apply still exists).** `applyAction(k_a, k_u, k_agent, k_report)` is a **receipt**, not the full gate. On-chain it requires (i) \(W[k_a,k_u]=1\), (ii) a nonzero commitment for that pair, (iii) the unit not already applied. It does **not** restore \(\tau\) or test \(u\in\mathrm{plan}(P)\). Those comparisons need the off-chain JSON. Software therefore evaluates, in order, whitelist, plan-binding (\(u\) and \(\tau\) identical to Reason’s unit), and digest (\(c_{\mathrm{chain}}=c_{\mathrm{rehash}}\)) **before** the receipt transaction. A still-whitelisted action copied from another cycle dies at plan-binding; a rewritten payload dies at the digest. Putting the full JSON on-chain would break the hash-only confidentiality claim. The split is intentional: the ledger freezes \(W\) and \(c\); Apply restores \(\mathrm{plan}_i\).

---

**Governance as a signal, not an actuator.** `TrustAnchored` is a tamper-evident event that a validated proposal exists. The SOC / SIEM / monitoring bus is a subscriber, not a vendor-specific product. Review/Approve from an operator is recorded back through the same registry and **re-enters** retrieve–compare; approval does not skip the gate. The contract freezes what may be done. It does not open API, gRPC, or MCP.

---

**Handoff to Execution and Monitoring.** Commit emits \(c\), the event, and the keys Apply will need to retrieve \(W[\hat{y}_i]\) and \(\mathrm{getCommitment}\). This closes the ungoverned-proposal failure that III-B left open: \(\mathrm{plan}_i\) can no longer be silently replaced by another fluent JSON. It does not close actuation. A still-whitelisted unit aimed at the wrong domain, or a second apply of the same \((k_{\mathrm{agent}},k_{\mathrm{report}},k_u)\), must be refused at the execution gate. Restoring \(\tau\), testing \(u\in\mathrm{plan}(P)\), and only then calling `applyAction` is the failure the Execution and Monitoring Layer exists to close.
