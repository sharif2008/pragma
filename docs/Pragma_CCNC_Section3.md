# III. END-TO-END: SYSTEM DESIGN

For trustworthy operational defense in a disaggregated enterprise network, we design a unified architecture with four static functional layers, as illustrated in Fig. 1: **(1) Detection**, where complementary observations at Access, Perimeter, and Endpoint are classified without pooling raw telemetry, and attribution names the evidence group that drove the decision; **(2) Reasoning and Planning**, where that detection product is turned into a policy-grounded, domain-specific mitigation plan; **(3) Blockchain Governance**, where the plan is cryptographically committed and the authorized action set is stored as an unpromptable whitelist; and **(4) Execution and Monitoring**, where an execution gate admits an action only if it is authorized for the predicted attack, identical to the committed plan, and integrity-checked against the on-chain digest.

Each network flow is processed in order through **Detect, Reason, Commit, and Apply**. Reason does not authorize execution. Commit does not dispatch. Apply is the only layer that may touch a live control plane, and only after the gates succeed.

---

## III-A. Detection Layer (Detect)

What fails without this layer: a centralized detector requires each enterprise domain to surrender raw telemetry to a single point, which is exactly the constraint that makes cross-domain detection impractical to begin with. Regulatory, ownership, and trust boundaries prevent the Access, Perimeter, and Endpoint layers of an enterprise from pooling raw features in one place, so detection must happen without centralizing evidence, or it does not happen at all in a realistic multi-party setting. We address this with a three-party Vertical Federated Learning (VFL) architecture: each party keeps its own columns, a local multilayer perceptron (MLP) maps that slice to an embedding, and only the embedding reaches a coordinator. Raw values never leave the party that owns them.

The layer has a second, equally necessary job. A label alone is not actionable for Section III-B: knowing the event is a PortScan does not tell the planner which domain must act first. After the VFL classifier is trained, KernelSHAP is run on a distilled surrogate of the fused representation so that \(\hat{y}_i\) is attributed to one of the three evidence groups. The pair \((\hat{y}_i,\, p_i^{\star})\) is the only object forwarded to Reason.

To show that locality is not an accuracy tax, the same split, loss, and optimizer are used to train a centralized MLP on the **concatenation of all features**. That baseline never appears on the live path; it exists so the VFL-versus-pooled comparison isolates feature locality rather than a mismatch in model capacity.

\[
x_i=\{x_i^1,x_i^2,x_i^3\}
\;\xrightarrow{\text{local MLP }f_p}\;
h_i^p\in\mathbb{R}^{64}
\;\xrightarrow{\text{concat}}\;
h_i\in\mathbb{R}^{192}
\;\xrightarrow{\text{MLP }g}\;
\hat{y}_i
\;\xrightarrow{\text{KernelSHAP}}\;
p_i^{\star}.
\]

**Feature partitioning.** Identifiers, addresses, and timestamps are dropped. The remaining 88 CICFlowMeter numeric columns are assigned to three evidence groups: Access / ISP (\(d_1=23\), volume/rate), Perimeter / IDS (\(d_2=32\), packet-shape and ports), Endpoint / EDR (\(d_3=42\), timing/direction). 97 listings over 88 unique columns; nine shared. Statistical proxy, not three sensors. Labels collapse to \(K=9\). Party-local StandardScaler; 64/16/20 stratified split, seed 42.

**Local MLP encoder.** \(d_p \rightarrow 128 \xrightarrow{\mathrm{ReLU},\,\mathrm{Dropout}(0.2)} 64 \xrightarrow{\mathrm{ReLU},\,\mathrm{Dropout}(0.1)}\). Only \(h_i^p\in\mathbb{R}^{64}\) leaves the party. Concatenation, not sum.

**Coordinator MLP.** \(h_i\in\mathbb{R}^{192} \rightarrow 128 \xrightarrow{\mathrm{ReLU},\,\mathrm{Dropout}(0.2)} K\). Logits; \(\mathrm{conf}_i=\max\mathrm{softmax}(z_i)\). Back-prop through concat; raw columns never move.

**Training.** Weighted CE (weights clamped \(20\times\)), Adam \(10^{-3}\) / \(10^{-4}\), grad clip \(1\), ReduceLROnPlateau (factor \(0.5\), patience 15), max 100 epochs, val every 5, early stop on macro-\(F_1\) (patience 20, \(\Delta_{\min}=0.001\)).

**Centralized all-feature baseline.** Concatenate all slices; MLP \(d\rightarrow 256\rightarrow 128\rightarrow 64\rightarrow K\), dropout \(0.2\), **same** recipe. Evaluation only; not deployed; does not feed Reason.

**Distilled SHAP.** Teacher VFL, student \(192\rightarrow 128\rightarrow 64\rightarrow K\), KL at \(T=3.0\) for 50 epochs. KernelSHAP on student (background 100, 200 perturbations). Sum \(\phi\) per 64-d block \(\Rightarrow p_i^{\star}\). Top-3 features per domain, top-5 overall for Reason.

**Handoff.** Emits \((\hat{y}_i, \mathrm{conf}_i, p_i^{\star}, \text{shares}, \text{top features})\). Evidence, not a plan: that is what III-B closes.

---

## III-B. Reasoning and Planning Layer (Reason)

Section III-A closes detection-without-centralization and forwards a structured record, not a narrative: the predicted class \(\hat{y}_i\), the detector confidence \(\mathrm{conf}_i\), the dominant evidence group \(p_i^{\star}\) obtained by summing KernelSHAP over each 64-dimensional fusion block, the three domain shares, and the top local features by \(\lvert\phi\rvert\). That record is still evidence. Knowing the event is a PortScan whose Perimeter block is largest does not name an authorized control, does not say whether Access or Endpoint should move second, and does not bind those choices to a catalog an execution gate can check. An unconstrained language model can invent a fluent mitigation that no playbook allows, or can tell a host-only story for a perimeter-dominated scan. We therefore treat Reasoning as a separate layer whose job is to convert the Detection product into one machine-checkable Mitigation Plans object. The layer does not authorize execution and does not write to the ledger; those failures belong to Commit and Apply.

The conversion is a fixed pipeline. After decode, nothing is free-form: the retrieval query is a filled template, ranking is a seven-step recall-then-precision cascade, the prompt is a slot-filled string, and the only accepted response is one JSON object whose action names must already appear in the per-attack catalog \(W[\hat{y}_i]\) that Commit will later enforce on-chain.

\[
(\hat{y}_i,\, p_i^{\star},\, \mathrm{conf}_i,\, \mathrm{SHAP}_i)
\;\xrightarrow{\text{query}}\;
q_i
\;\xrightarrow{\text{retrieve + rank}}\;
\mathcal{R}_i
\;\xrightarrow{\text{prompt}}\;
\pi_i
\;\xrightarrow{\text{JSON}}\;
\mathrm{plan}_i.
\]

**From Detection into a query.** Domain names in every later string are forced to exactly Access / ISP, Perimeter / IDS, or Endpoint / EDR — the same three evidence groups used to split the 88 CICFlowMeter columns and to aggregate SHAP. \(\hat{y}_i\) selects the allowed-action list \(W[\hat{y}_i]\). \(\mathrm{conf}_i\) is written into both the query and the task block so intensity (`Immediate` … `Low`) tracks the detector rather than the model’s tone. \(p_i^{\star}\) becomes the sentence “Dominant network domain: … (contribution: \(x\%\))”; top-3 features per domain become retrieval keywords, and only the top-5 features by \(\lvert\phi\rvert\) are admitted into the evidence JSON the model may cite. Retrieval does not ask the language model to invent a search string. The operational query is a single template filled from that record. Optional rephrases exist; the evaluated path uses only the template, so two identical Detection records produce the same \(q_i\). That determinism is required by the next layer: Commit hashes a plan that must be reproducible from the same evidence.

**Policy index.** The corpus is closed — NIST SP 800-53, NIST SP 800-207, CIS Controls, MITRE ATT&CK, and the local response catalog — not an open web. Documents are stored parent/child. A parent is a semantic section (the text the planner will read). A child is a short passage used only for nearest-neighbor search (size 384, overlap 96, embedding `all-MiniLM-L6-v2`), tagged with \((\mathrm{parent\_id},\, \mathrm{child\_index},\, \mathrm{source})\). Ranking operates on children; the prompt receives expanded parents.

**Retrieve-then-rank cascade.** Dense search alone is the wrong ranking: a bi-encoder returns topical neighbors and, left unchecked, fills the prompt with near-duplicates from one source. Each step closes a failure of the previous one. (1) *Dense retrieve:* FAISS oversamples (\(\min(300,\, 20\times 6)\)), keeps 20 children, converts distance \(d\) to \(1/(1+d)\), and balances across source files. (2) *Merge and dedupe* by parent/child identity so overlap does not starve the later budget. (3) *MMR* with \(\lambda=0.5\) keeps up to 60 children, \(\mathrm{MMR}(c)=\lambda\cdot\mathrm{sim}(q,c)-(1-\lambda)\cdot\max_{c'\in S}\mathrm{sim}(c,c')\), so NIST, CIS, and ATT&CK can coexist. (4) *Cross-encoder* `ms-marco-MiniLM-L-6-v2` scores each \((q,\mathrm{passage})\) jointly (passage truncated at 4\,000 characters) — vector similarity is a neighborhood, not “this passage mitigates *this* PortScan at the perimeter.” It is too expensive for the whole index, which is why (1)–(3) exist. (5) Keep the top 20 children. (6) Expand each to its parent (12\,000-character cap); duplicate parents collapse. (7) Emit the top five numbered sections, or the explicit empty-KB sentence. Retrieval returns guidance; it does not invent action names. \(W[\hat{y}_i]\) is a different prompt slot and the same catalog Commit will store on-chain.

**Prompt construction.** The user message is one template filled in software before decode. The model never sees raw telemetry, never sees the 192-d fusion vector, and never sees names outside \(W[\hat{y}_i]\). Slots are: role and the three exact domain strings; prediction summary; the dominance sentence from \(p_i^{\star}\); Detection evidence JSON (top-5 features only); \(W[\hat{y}_i]\) plus preferred domains and evidence cues; a per-domain card (role, top-3 features, actions that domain may execute); numbered RAG sections or the empty-KB sentence. The task block requires interpretation on all three domains, a first-mover domain, catalog-only actions, a `network_tier` on each unit, evidence-tied reasoning, priority scaled by \(\mathrm{conf}_i\), and empty lists rather than a made-up control. Decoding uses GPT-4o-mini at \(T=0.3\); the first `{` through the last `}` is parsed.

**Response schema.** The only accepted output is \(\mathrm{plan}_i\): `threat_level` \(\in\) {Critical, High, Medium, Low}; `primary_actions` / `supporting_actions` as `{action, network_tier, party_evidence_type, reasoning}`; `all_actions` their union; `overall_reasoning`; `execution_priority` \(\in\) {Immediate, High, Standard, Low}; `knowledge_sources_used`. `action` is copied from \(W[\hat{y}_i]\) (`limit rate` remains `limit rate`). `network_tier` is exactly one of the three domain strings. Primary actions prefer \(p_i^{\star}\) unless the evidence contradicts it. A PortScan whose Perimeter share dominates is planned as, for example, `tarpit scan` and `harden ports` at Perimeter / IDS with supporting `update ACL` at Access / ISP — names legal for PORTSCAN and illegal for BENIGN. Apply will reject any name absent from \(W[\hat{y}_i]\) or absent from this object.

**Handoff to Blockchain Governance.** \(\mathrm{plan}_i\) is this layer’s only output. It is the off-chain payload \(P\) that Section III-C will canonicalize and commit as \(c=\mathrm{SHA256}(P)\). Reason never calls the contract and never opens a device API. This closes the evidence-to-plan failure that III-A left open. It does not close the trust failure that follows generation. A fluent, catalog-legal action copied from another cycle, a rewritten `network_tier`, or a payload whose bytes change before dispatch will still look like a valid plan if the next layer only stores text. Binding \(W[\hat{y}_i]\) on-chain, writing \(c\) once, and refusing any apply that does not re-hash to \(c\) is the failure the Blockchain Governance Layer exists to close.

---

## III-C. Blockchain Governance Layer (Commit)

Section III-B forwards a single object \(\mathrm{plan}_i\)---the Mitigation Plans JSON whose actions were selected from \(W[\hat{y}_i]\) and assigned to Access, Perimeter, or Endpoint. What fails without this layer is that \(\mathrm{plan}_i\) is still a proposal. Reasoning is probabilistic. A language model can emit a fluent action that is still harmful on a live path, and a poisoned or substituted plan can change between generation and dispatch. A policy document in a prompt is not a control: it can be ignored, reworded, or overwritten. Autonomous defense therefore needs a pre-execution governance layer that cannot hallucinate mitigations and cannot be talked into accepting them.

A smart contract is a program that lives on a ledger and executes exactly the rules written in its code. Callers submit transactions; the network orders them; every honest replica applies the same function to the same state. We use that property for two jobs: an unpromptable action whitelist, and a write-once commitment to the plan produced for this detection cycle.

**Policy whitelist and plan anchoring.** The registry is seeded with a per-attack map \(W[a,u]\) drawn from the same constrained catalog the planner is allowed to name. An action that is legal for DDoS but not for the predicted class fails the whitelist even if the language of the plan is fluent. After the Mitigation Plans object is persisted off-chain, the backend canonicalizes payload \(P\) (identity, retrieval context, structured plan, and assigned domains) and writes only

\[
c = \mathrm{SHA256}(P)
\]

on-chain. Telemetry, SHAP tables, and raw model text never enter the ledger. Overwrites of an existing commitment are rejected, so \(c\) is write-once. Later readers recover the digest and compare it to a re-hash of the stored payload. Unauthorized edits are detected because the hashes no longer match; confidentiality is preserved because the ledger holds a digest, not the data.

Immutability in this architecture is therefore not “the blockchain stores the network.” It is three enforceable facts: the allowed action set for each attack is on-chain and not promptable; the plan for this cycle has a single committed digest; and any later apply must bind to both.

**Governance as a signal, not an actuator.** Once a plan is anchored, the commitment event is a tamper-evident signal that a validated proposal exists. That signal can be consumed by whatever operational fabric the enterprise already runs: a SOC console, a SIEM correlation rule, a ticket queue, or an equivalent monitoring bus. The architecture does not require a vendor-specific SOC. The contract freezes what may be done; it does not dispatch.

This closes the ungoverned-proposal problem, but a frozen plan is still not an actuation. A still-whitelisted action copied from another cycle, or a unit whose domain no longer matches the plan, must be refused at the point of execution. Restoring that check is the failure the next layer exists to close.

---

## III-D. Execution and Monitoring Layer (Apply)

What fails without this layer: Commit records that a plan existed, but recording is not enforcement. Without an execution gate, an orchestrator could still dispatch a different legal action, a silently edited payload, or a unit aimed at the wrong enterprise domain. Apply is the only place a planned action may touch the live network. After the SOC or core monitoring path is notified, an event listener retrieves the on-chain record, the gate compares that record with policy, and only then may the orchestrator dispatch.

**Execution gate.** The gate does not invent mitigations. It compares three facts that already exist: (i) whether action \(u\) is on the whitelist \(W\) for this attack; (ii) whether \(u\) and its domain \(\tau\) (Access, Perimeter, or Endpoint) are the same actions Reason wrote into plan \(P\); (iii) whether a re-hash of the off-chain payload still matches the on-chain digest. In compact form,

\[
W[a,u]=1 \;\wedge\; u\in\mathrm{plan}(P) \;\wedge\; c_{\mathrm{chain}}=c_{\mathrm{rehash}}.
\]

Software evaluates the three checks in that order — whitelist, then plan-binding, then digest — before the registry is asked to record an apply. The on-chain apply then re-checks that the action is still whitelisted, that a commitment exists, and that the same unit has not already been applied. An action that is only plausible, or that is legal for a different attack, fails the whitelist. A still-whitelisted action copied from another cycle fails plan-binding. A silently edited payload fails the digest check.

**Per-attack whitelist.** Labels follow the CICIDS-2017 grouping used by Detect. Actions are a constrained catalog, not free-form model text:

| Attack \(a\) | Allowed actions (excerpt) |
|--------------|---------------------------|
| BENIGN | log incident, monitor traffic |
| DDOS / DOS | limit rate, enable scrubbing, block IP, enable syncookies, update ACL |
| SSHPATATOR / FTPPATATOR | fail2ban block, lock account, enforce MFA |
| PORTSCAN | tarpit scan, harden ports, update ACL |
| WEBATTACK | apply WAF, virtual patch, isolate service |
| BOT | captcha challenge, reputation filter |
| OTHERS | log incident, isolate service, update ACL |

**Reject path and human loop.** If the gate does not match, the plan is marked rejected or blocked. The flow goes to operator review, not to the network. Every such outcome is written to the same monitoring archive as a successful apply, so a denial is as visible as an actuation. If an operator later reviews and approves, that approval is recorded back through the governance layer and re-enters the same retrieve–compare path. Approval does not skip the gate.

**Success path.** If the gate validates the plan, control passes to the execution agent. That agent invokes the domain-specific interface — API, gRPC, or MCP — for the Access, Perimeter, or Endpoint tier named in the plan. A successful unit is recorded on-chain and refused if presented again. The evaluated executor records receipts rather than driving a live device; a production deployment can replace that stub with the same control-plane fabric that collected the flow. Dispatch is therefore a signed, gated call, not a free-form command bus.

**Commit and Apply.**

```
COMMIT(agentKey, reportKey, c)
    require c ≠ 0 and not already committed
    store commitment c
    emit trust-anchored

APPLY(a, u, τ, agentKey, reportKey)
    require τ matches the domain written in plan(P)
    require W[a, u] = 1
    require a commitment exists and has not already been applied for this unit
    require u ∈ plan(P) and c_chain = SHA256(P)
    mark applied; emit action-applied
    dispatch u to domain τ
```

Every hop — notify, retrieval, whitelist miss, plan mismatch, human approval, successful apply — emits an audit record back to the monitoring core. The core archive is the operator-visible history; the ledger holds the immutable receipts. This layer guarantees that only a whitelisted, bound, integrity-checked action reaches an agent. It does not claim that the language-model plan was correct.

---

*End of Section III.*
