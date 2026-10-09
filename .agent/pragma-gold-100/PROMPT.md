# Prompt: pragma-gold-100 (Muse Spark)

Paste the **Muse Spark block** into a Cursor Task (`subagent_type=generalPurpose`, `model=muse-spark-1.3-high`).

Do **not** use `gpt-6-luna` / `OPENAI_API_KEY`. That model is reserved for `pragma-rag-eval100`.

Parent agent (this chat) must finish **scripts first**: reconstruct `test_idx`, write **10 flows per class (N=90)** to `flows.csv`, run `detect_predict` on those 90 rows. Then launch Muse Spark with the detect JSON path filled in.

---

## Parent (scripts only — not Muse Spark)

```text
Repo: D:\Projects\ChainAgentVFL
Follow .agent/pragma-gold-100/SPEC.md and PLAN.md.

1. Reconstruct VFL test split from the same CSV as last detect-train (random_state=42, 20% stratified). Save indices.
2. From test_idx only (seed 42), take **exactly 10 rows per class** (N=90). Nine labels: BENIGN, BOT, DDOS, DOS, FTPPATATOR, OTHERS, PORTSCAN, SSHPATATOR, WEBATTACK. Include BENIGN. No Heartbleed/Infiltration.
3. Write experiments/fixtures/gold-100/flows.csv (not reason_ablation_9). Treat split_index as reserved.
4. Run detect_predict on exactly those 90 rows (KernelSHAP). Do not use the 9-row ablation fixture as gold source.
5. Freeze gold that is PDF-grounded (parent body quotes + W[class] action the control supports). Muse Spark if available; else gold_freeze.py. Do not call gpt-6-luna / OPENAI_API_KEY (reserved for pragma-rag-eval100).
```

---

## Muse Spark (paste this)

```text
You are drafting gold labels for PRAGMA RAG/LLM eval. You are NOT the authority. Catalogs + ten PDFs + parent-agent checks are.

Repo: D:\Projects\ChainAgentVFL
Read first: .agent/pragma-gold-100/SPEC.md

FILL IN (parent agent):
- DETECT_JSON: <experiments/gold-100/predictions_detailed_*.json for the 90 gold rows>
- DATE: <YYYY-MM-DD>

Read-only inputs (do not invent fields):
- DETECT_JSON (true_label, predicted_label, confidence, SHAP, domain shares, row / split_index)
- backend/storage/attack_options.json  → attacks[TRUE_LABEL], evidence_cues, primary_domains
- backend/storage/agentic_features.json → domain names, action_capabilities
- experiments/rag-index/vector_store/rag_parents.json → parent_id, source_file, section_heading, text
- experiments/fixtures/gold-100/flows.csv

Closed knowledge filenames (source_file must be exact; no 11th PDF, no SOC 2, no FiGHT, no web):
NIST-SP-800-53-Rev5-Security-Privacy-Controls.pdf
NIST-SP-800-207-Zero-Trust-Architecture.pdf
ISO-IEC-27001-ISMS-Requirements.pdf
CIS-Controls-v8.1.pdf
MITRE-ATTCK-Design-and-Philosophy.pdf
MITRE-ATTCK-Building-Better-Defenses.pdf
CISO-Guide-Top-Cybersecurity-Frameworks.pdf
Open-RAN-Security-Report.pdf
NIST-SP-800-53-vs-ISO-IEC-27001-Comparison.pdf
Review-NIST-ISO27001-HIPAA-MITRE-ATTCK.pdf

Action vocabulary — copy catalog spelling only (never BLOCK_IP, RATE_LIMIT_IP, ALERT_SOC, HUMAN_APPROVAL, NO_ACTION, ISOLATE_HOST):
BENIGN: log incident, monitor traffic
DDOS: limit rate, enable scrubbing, blackhole route, connection limit, enable syncookies, block IP, update ACL, scale service
DOS: limit rate, enable syncookies, connection limit, enable scrubbing, block IP, update ACL, monitor traffic
SSHPATATOR: fail2ban block, lock account, throttle credentials, enforce MFA, block IP, update ACL, connection limit, monitor traffic
FTPPATATOR: fail2ban block, lock account, throttle credentials, enforce MFA, block IP, update ACL, connection limit, isolate service
PORTSCAN: block IP, tarpit scan, scan threshold, harden ports, update ACL, limit rate, reputation filter, log incident
WEBATTACK: apply WAF, virtual patch, block IP, limit rate, update ACL, isolate service, reputation filter, monitor traffic
BOT: captcha challenge, js challenge, reputation filter, limit rate, apply WAF, block IP, update ACL, monitor traffic
OTHERS: log incident, monitor traffic, limit rate, update ACL, block IP, isolate service, enable scrubbing

network_tier is exactly: "Access / ISP" | "Perimeter / IDS" | "Endpoint / EDR"
Prefer primary_domains[TRUE_LABEL] when it agrees with SHAP-dominant domain.

Per case:
- primary_action ∈ W[true_label]
- acceptable_actions ⊆ W[true_label] and includes primary (small subset)
- unsafe_actions ⊆ (union of all W[*] minus acceptable); must not contain primary
- BENIGN primary ∈ {log incident, monitor traffic}; unsafe includes block/rate/WAF-style strings from the global union that must not fire on BENIGN

Rationale + 2–5 atomic_reasoning_points may use ONLY:
1. this row’s true_label + flow/SHAP
2. W[true_label], evidence_cues, primary_domains
3. a quoted span of listed relevant_rag_chunks parent *body text* (not heading)

primary_action must be the W[true_label] string the cited PDF control supports (CIS 8→log incident, CIS 10→reputation filter, CIS 13/DoS→limit rate or block IP, CIS 5/6→lock account or enforce MFA, CIS 16→apply WAF, CIS 17→log incident). Do not default to W[0] if the PDF supports another whitelist action.

Do not add ATT&CK technique ids, product names, or controls unless they appear in those parents or catalogs.

RAG gold: search ONLY rag_parents.json (source_file ∈ the ten PDFs). Keep 1–3 parents whose text actually supports the class or chosen control. Store parent_id, source_file, section_heading, text_hash = "sha256:" + SHA-256 hex of that parent’s text as stored. If none support the case, relevant_rag_chunks = [] and the rationale must say the corpus has no matching section. Do not mint ids like MITRE-ATTACK-TA0043-CHUNK-07. Do not run reason.py / eval A/B/C to pick gold chunks.

case_id: G-001 … G-090 in flows.csv order. flow_id from the CSV row id or a stable hash. split_index must be the test-split index from split_manifest / detect JSON.

Write 90 cases (10 per class). Schema for each object:

{
  "case_id": "G-001",
  "flow_id": "<csv row id or hash>",
  "split_index": 0,
  "true_label": "PORTSCAN",
  "predicted_label": "PORTSCAN",
  "confidence": 0.0,
  "flow_evidence": {
    "shap_features": [],
    "dominant_domain": "Perimeter / IDS",
    "domain_shares": {}
  },
  "ground_truth": {
    "primary_action": "block IP",
    "primary_network_tier": "Perimeter / IDS",
    "acceptable_actions": ["block IP", "update ACL", "log incident"],
    "unsafe_actions": ["enable scrubbing", "scale service"]
  },
  "gold_rationale": "...",
  "atomic_reasoning_points": ["...", "..."],
  "relevant_rag_chunks": [
    {
      "parent_id": "p_030d22532d57a27d",
      "source_file": "CIS-Controls-v8.1.pdf",
      "section_heading": "",
      "text_hash": "sha256:…"
    }
  ]
}

Outputs (draft under drafts/, then freeze — same bytes, same SHA-256):

experiments/gold-100/
  drafts/                 # raw draft JSON only here
  gold_security_cases.json
  gold_security_cases.csv
  split_manifest.json     # dataset path, CSV SHA-256, seed, split lengths, selected split_index list, class counts
  generation.json
  README.md
  SHA256.txt

generation.json exactly:
{
  "model_name": "Muse Spark",
  "model_source": "cursor-subagent",
  "model_slug": "muse-spark-1.3-high",
  "openai_api_used": false,
  "eval_model_reserved": "gpt-6-luna",
  "date": "<DATE>"
}

Then copy the freeze (not drafts/) to data/ground_truth/ with identical SHA256.txt.

Stop. Do not edit after hash. Do not run BERTScore / RAG eval / Commit / Apply. Do not call OpenAI.
If a parent_id is missing from rag_parents.json, drop that citation. Empty list is valid; a fake id is not.
```

---

## After Muse Spark (parent validates)

Before treating gold as frozen, check SPEC quality gates:

- 90 unique `case_id` and `split_index`, all in reserved fixture `selected_split_index`
- each cited parent body supports the class; rationale quotes that body
- every action string ∈ `attack_options.json`
- every `network_tier` ∈ the three domains
- every `parent_id` ∈ `rag_parents.json`; `text_hash` matches parent text
- BENIGN primaries only `log incident` or `monitor traffic`
- `generation.json`: Muse Spark, `openai_api_used: false`
- `experiments/gold-100/` and `data/ground_truth/` share `SHA256.txt`
