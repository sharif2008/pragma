# Outstanding Reviewer Feedback — Pragma / CCNC Paper

Status as of current draft. Split into what's already closed and what a reviewer would still flag.

## Already Closed

- **Threat model** — explicit adversary-considered / not-considered / scoped-guarantee subsection now sits before the mechanism it justifies (Section III-C, between Reasoning and Blockchain Governance).
- **"Why blockchain vs. signed hash"** — now answered directly in III-D rather than assumed.
- **Surrogate-SHAP disclosure** — now stated honestly: KernelSHAP runs on a distilled meta-model trained by soft-label knowledge distillation, not on the VFL classifier directly, with fidelity-check against the VFL model described.
- **VFL partition artificiality** — disclosed as a controlled statistical proxy for feature-level data ownership, not presented as real cross-domain separation.
- **Section III connective structure** — each layer states what fails without it, so the architecture reads as an argument rather than a components list.

## Still Open

1. **No precision/recall breakdown for Table V.**
   Macro-F1 (0.86) vs. macro-recall (0.99) implies a precision problem somewhere. Not yet reported per-class. Near-certain reviewer question for any detection-performance table.

2. **Single-seed VFL-vs-centralized comparison.**
   The +0.0016 macro-F1 "improvement" is noise at one seed. Needs either multiple seeds with variance reported, or an explicit reframe to "comparable, not better."

3. **Tamper experiment is still self-confirming.**
   Table VI's 100% block rate is guaranteed by construction — the tampering injected is exactly what the gates check for. Two fixes:
   - Log natural gate rejections against the LLM's own unconstrained proposals (not just injected tampering) — shows the gate catching real model behavior, not a synthetic adversary.
   - Confirm Section V explicitly states this experiment validates gate correctness against the scoped adversary from III-C, not general robustness (draft sentence exists — verify it's actually in the manuscript).

4. **No plan-quality evaluation.**
   Nothing currently measures whether GPT-4o-mini's mitigation plans are good — only that non-compliant ones get blocked. Most likely to draw a "does the LLM add value" question, especially at journal tier.

5. **Whitelist table consistency check needed.**
   Newer whitelist draft (e.g., "enable syncookies," "captcha challenge, reputation filter") differs from the original Table III entries used in the Section V experiment. If Table VI's counts were computed against the older catalog, this is a silent inconsistency between what Section III describes as policy and what Section V tested. Needs a direct verification pass.

6. **Single local Hardhat node for all latency numbers.**
   Table VII's ~79ms commit+verify won't be read as representative of a production permissioned chain. No multi-node run yet.

7. **New citations need verification.**
   `lundberg2017shap` and `hinton2015distill` were added as new bib entries without search access — confirm against a real source before the bibliography is final. Also verify `wu2023falcon` actually supports the surrogate-SHAP claim it's attached to, rather than being only topically adjacent.

8. **Typos / formatting pass.**
   "concensus," stray cross-reference to "Table I" where "Table III" is meant, missing spaces after periods, etc. Cheap fixes, still worth a final pass before submission.

## Priority Ranking (impact per effort)

| Priority | Item | Why |
|---|---|---|
| 1 | #1, #2 | Cheapest fixes, caught in first five minutes of reading Table V |
| 2 | #5 | Verification only, not new work — do before anything else ships |
| 3 | #3, #4 | Largest effect on journal- vs. conference-tier acceptance, but require new experimentation |
| 4 | #6, #7, #8 | Lower cost, lower individual impact, but expected by a careful reviewer |
