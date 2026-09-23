# III-B. Reasoning and Planning Layer (Reason)

Section III-A closes detection-without-centralization and forwards a structured record, not a narrative: the predicted class \(\hat{y}_i\), the detector confidence \(\mathrm{conf}_i\), the dominant evidence group \(p_i^{\star}\) obtained by summing KernelSHAP over each 64-dimensional fusion block, the three domain shares, and the top local features by \(\lvert\phi\rvert\). That record is still evidence. Knowing the event is a PortScan whose Perimeter block is largest does not name an authorized control, does not say whether Access or Endpoint should move second, and does not bind those choices to a catalog an execution gate can check. An unconstrained language model can invent a fluent mitigation that no playbook allows, or can tell a host-only story for a perimeter-dominated scan. We therefore treat Reasoning as a separate layer whose job is to convert the Detection product into one machine-checkable Mitigation Plans object. The layer does not authorize execution and does not write to the ledger; those failures belong to Commit and Apply.

The conversion is a fixed pipeline. After decode, nothing is free-form: the retrieval query is a filled template, ranking is a seven-step recall-then-precision cascade, the prompt is a slot-filled string, and the only accepted response is one JSON object whose action names must already appear in the per-attack catalog \(W[\hat{y}_i]\) that Commit will later enforce on-chain.

\[
\underbrace{(\hat{y}_i,\, p_i^{\star},\, \mathrm{conf}_i,\, \mathrm{SHAP}_i)}_{\text{III-A output}}
\;\xrightarrow{\text{query}}\;
q_i
\;\xrightarrow{\text{retrieve + rank}}\;
\mathcal{R}_i
\;\xrightarrow{\text{prompt}}\;
\pi_i
\;\xrightarrow{\text{JSON}}\;
\underbrace{\mathrm{plan}_i}_{\text{III-C payload } P}.
\]

**From Detection into a query.** Domain names in every later string are forced to exactly Access~/~ISP, Perimeter~/~IDS, or Endpoint~/~EDR---the same three evidence groups used to split the 88 CICFlowMeter columns and to aggregate SHAP. \(\hat{y}_i\) selects the allowed-action list \(W[\hat{y}_i]\) and the preferred-response domains for that class. \(\mathrm{conf}_i\) is written into both the query and the task block so intensity (`Immediate` \(\ldots\) `Low`) tracks the detector rather than the model's tone. \(p_i^{\star}\) and the three block shares become the sentence “Dominant network domain: \(\ldots\) (contribution: \(x\%\))”; top-3 features per domain become retrieval keywords, and only the top-5 features by \(\lvert\phi\rvert\) are admitted into the evidence JSON the model may cite. Retrieval does not ask the language model to invent a search string. The operational query is a single template filled from that record---class, confidence, dominant domain, and the three keyword lists---asking for SOC / NIST / CIS / ATT\&CK controls appropriate to that confidence. Optional rephrases exist; the evaluated path uses only the template, so two identical Detection records produce the same \(q_i\). That determinism is required by the next layer: Commit hashes a plan that must be reproducible from the same evidence.

**Policy index.** The corpus is closed---NIST SP 800-53, NIST SP 800-207, CIS Controls, MITRE ATT\&CK, and the local response catalog---not an open web. Documents are stored parent/child. A parent is a semantic section (the text the planner will read). A child is a short passage used only for nearest-neighbor search (size 384, overlap 96, embedding model `all-MiniLM-L6-v2`), tagged with \((\mathrm{parent\_id},\, \mathrm{child\_index},\, \mathrm{source})\). Ranking operates on children; the prompt receives expanded parents. A child is precise enough to retrieve; a parent is long enough to be a control rather than a fragment that lost its if/then scope.

**Retrieve-then-rank cascade.** Dense search alone is the wrong ranking for this job. A bi-encoder returns topical neighbors and, left unchecked, fills the prompt with near-duplicates from one source while dropping a short CIS or ATT\&CK passage that is more actionable. Each step exists to close a failure of the previous one.

1. *Dense retrieve (recall).* FAISS returns an oversampled neighbor list (\(\min(300,\, 20\times 6)\)), keeps 20 children, converts distance \(d\) to similarity \(1/(1+d)\), and balances hits across source files so one document cannot occupy the entire top-20. The bi-encoder is cheap and high-recall; it is not a relevance judge.
2. *Merge and dedupe.* Multi-query children are unioned by \((\mathrm{parent\_id},\, \mathrm{child\_index})\) (else source, title, and a short prefix). The higher similarity wins. Without this step, overlap would inject the same passage twice and starve the later budget.
3. *MMR (diversity).* Maximum Marginal Relevance with \(\lambda=0.5\) keeps up to 60 children, \(\mathrm{MMR}(c)=\lambda\cdot\mathrm{sim}(q,c)-(1-\lambda)\cdot\max_{c'\in S}\mathrm{sim}(c,c')\). After merge the pool is still redundant; without MMR the cross-encoder would re-score the same control and NIST, CIS, and ATT\&CK would not coexist in the prompt.
4. *Cross-encoder (precision).* Each surviving \((q,\mathrm{passage})\) pair is scored jointly by `cross-encoder/ms-marco-MiniLM-L-6-v2` (passage truncated at 4\,000 characters) and sorted. Vector similarity is a topical neighborhood, not “this passage is a mitigation for *this* PortScan at the perimeter.” The cross-encoder is the relevance model; it is too expensive to run on the whole index, which is why steps 1--3 exist.
5. *Rerank cut.* The top 20 children by cross-encoder score are kept.
6. *Parent expansion.* Each kept child is replaced by its parent section (capped at 12\,000 characters); duplicate parents collapse to the higher child score. The planner must quote a control, not a 384-character shard.
7. *Prompt budget.* The top five sections (up to ten) are emitted as numbered `[k] title + body + source`. If the list is empty, the slot is the explicit sentence that no relevant documents were found, and the model is forbidden to invent policy.

Retrieval returns guidance and allowable primitives. It does not invent action names. The closed list \(W[\hat{y}_i]\) is supplied in a different prompt slot and is the only legal vocabulary for `action`---the same catalog the Blockchain Governance Layer will store as an unpromptable whitelist.

**Prompt construction.** The user message is one template whose slots are filled in software before any decode. The model never sees raw telemetry, never sees the 192-dimensional fusion vector, and never sees action names outside \(W[\hat{y}_i]\). The slots are: (i)~role and the three exact domain strings; (ii)~prediction summary (\(\hat{y}_i\), \(\mathrm{conf}_i\)); (iii)~the dominance sentence produced from \(p_i^{\star}\); (iv)~the Detection evidence JSON, top-5 features only; (v)~\(W[\hat{y}_i]\) plus preferred domains and typical evidence cues for that class; (vi)~a per-domain card (role, top-3 features, actions that domain is allowed to execute---the intersection of class whitelist and domain capability); (vii)~the numbered RAG sections, or the empty-KB sentence. The task block, also template text, requires the model to interpret \(\hat{y}_i\) on all three domains, name which domain must act first, select actions only from the list, assign each a `network_tier`, cite evidence in per-action reasoning, scale `execution_priority` with \(\mathrm{conf}_i\), and return empty lists rather than a made-up control if nothing fits. Decoding uses GPT-4o-mini at temperature \(T=0.3\) with the instruction to return only valid JSON; the first `{` through the last `}` is parsed and surrounding prose is discarded.

**Response schema.** The only accepted output is one object \(\mathrm{plan}_i\):

```
threat_level        ∈ {Critical, High, Medium, Low}
all_actions         = union of primary and supporting names
primary_actions[]   = {action, network_tier, party_evidence_type, reasoning}
supporting_actions[]= same shape
overall_reasoning   = why this Access / Perimeter / Endpoint order
execution_priority  ∈ {Immediate, High, Standard, Low}
knowledge_sources_used
```

`action` is copied from \(W[\hat{y}_i]\), not paraphrased (`limit rate` remains `limit rate`). `network_tier` is exactly one of the three domain strings. Primary actions prefer \(p_i^{\star}\) unless the evidence contradicts it. Empty RAG forbids invented policy citations. A PortScan whose Perimeter share dominates is therefore planned as, for example, `tarpit scan` and `harden ports` at Perimeter~/~IDS with a supporting `update ACL` at Access~/~ISP---names that are legal for PORTSCAN and illegal for BENIGN. That legality is not left to the model: Apply will reject any name that is absent from \(W[\hat{y}_i]\) or absent from this object.

**Handoff to Blockchain Governance.** \(\mathrm{plan}_i\) is this layer's only output. It is the off-chain payload \(P\) that Section III-C will canonicalize and commit as \(c=\mathrm{SHA256}(P)\). Reason never calls the contract and never opens a device API. This closes the evidence-to-plan failure that III-A left open: the planner cannot speak an action that is not in the catalog, cannot omit a domain assignment, and cannot hide the Detection fields it used. It does not close the trust failure that follows generation. A fluent, catalog-legal action copied from another cycle, a unit whose `network_tier` is rewritten after decode, or a payload whose bytes change before dispatch will still look like a valid plan if the next layer only *stores* text. Binding \(W[\hat{y}_i]\) on-chain, writing \(c\) once, and refusing any apply that does not re-hash to \(c\) is the failure the Blockchain Governance Layer exists to close.
