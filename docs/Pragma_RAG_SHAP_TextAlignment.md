# Retrieval-Augmented Mitigation Text vs Gold Policy (No RAG vs RAG + SHAP)

**PRAGMA evaluation note.** Off-gold hybrid eval, \(N=200\) flows, systems A (`LLM_only`) vs B (`RAG_RANKING`). Figures are 3D renderings of `experiments/rag/hybrid_eval200/comparison.csv`. Live copies: `bertscore_rouge_3d.png` (headline) and `bertscore_rouge_by_class_3d.png` (nine attack types). Complements the 2D grouped bars in `bertscore_rouge.png`.

---

## Abstract

On 200 CICIDS-style flows held out from gold-100, ranked hybrid retrieval plus SHAP raises overlap with the gold policy passages on every flow and every metric. Mean BERTScore F1 moves from 0.774 to 0.817 (\(+0.043\)). ROUGE-1 more than doubles, 0.138 to 0.333 (\(+0.195\)). ROUGE-L doubles, 0.068 to 0.139 (\(+0.071\)). B exceeds A in **200/200** cases for BERTScore F1, ROUGE-1, and ROUGE-L. The lift is present in all nine labels; ROUGE-1 is the largest absolute gain, BERTScore the most uniform.

## Keywords

Retrieval-augmented generation, BERTScore, ROUGE, SHAP, hybrid FAISS+BM25, CICIDS-2017, mitigation planning

---

## I. Setup

**Fixture.** `experiments/data/eval-200/` subset of `rag_reason_500`, disjoint from gold-100 (`split_index` overlap \(=0\)). Mix: 23 BENIGN, 23 BOT, and 22 each of DDOS, DOS, FTPPATATOR, OTHERS, PORTSCAN, SSHPATATOR, WEBATTACK. Model `gpt-6-luna`, \(T=0\). 400 Mitigation Plans.

**Systems.** A is LLM-only (no retrieve, no SHAP). B is hybrid retrieve then rank: FAISS \(N=80\) and BM25 \(N=80\) on the same child ids, RRF, MMR \(\lambda=0.5\), 20 children expanded to 5 parent sections, plus SHAP in the prompt. One LLM call per flow per system.

**Reference.** Gold-100 `relevant_rag_chunks[].text` concatenated **per attack type**. Hypothesis A is the rationale only. Hypothesis B is the five prompt parents plus the rationale. This measures policy-text grounding, not catalog-action match.

**Source file.** Means below are the column averages of `comparison.csv` (`A_*` = No RAG, `C_*` = RAG + SHAP). Rounded to three decimals to match the 2D headline chart.

---

## II. Headline (3D)

**Figure 1.** `bertscore_rouge_3d.png`. Metrics on \(x\), system on \(y\), score on \(z\). Grey = No RAG. Blue = RAG + SHAP. Table under the axes is the same mean as Figure 1 of the 2D report.

| Metric | No RAG (A) | RAG + SHAP (B) | B − A | B / A |
|--------|-----------:|---------------:|------:|------:|
| BERTScore F1 | 0.774 | 0.817 | +0.043 | \(1.06\times\) |
| ROUGE-1 | 0.138 | 0.333 | +0.195 | \(2.41\times\) |
| ROUGE-L | 0.068 | 0.139 | +0.071 | \(2.04\times\) |

B is closer to the gold class concat than A on all 200 flows for all three metrics (0 ties). ROUGE-1 is the headline lexical gain because B copies parent control language that A never sees. BERTScore already sits high for A (shared SHAP/catalog English), so the semantic lift is smaller but strictly one-sided.

---

## III. By attack type (3D)

**Figure 2.** `bertscore_rouge_by_class_3d.png`. Three panels (BERTScore F1, ROUGE-1, ROUGE-L). \(x\) = label, \(y\) = system, \(z\) = mean score over that label’s \(n\).

**Table I.** Mean BERTScore F1 by true label.

| Attack | \(n\) | No RAG | RAG + SHAP | B − A |
|--------|--:|-------:|-----------:|------:|
| BENIGN | 23 | 0.782 | 0.826 | +0.044 |
| BOT | 23 | 0.774 | 0.808 | +0.034 |
| DDOS | 22 | 0.771 | 0.816 | +0.045 |
| DOS | 22 | 0.769 | 0.818 | +0.049 |
| FTPPATATOR | 22 | 0.773 | 0.815 | +0.042 |
| OTHERS | 22 | 0.783 | 0.824 | +0.041 |
| PORTSCAN | 22 | 0.770 | 0.805 | +0.036 |
| SSHPATATOR | 22 | 0.770 | 0.819 | +0.049 |
| WEBATTACK | 22 | 0.773 | 0.821 | +0.049 |
| Overall | 200 | 0.774 | 0.817 | +0.043 |

Smallest BERTScore lift is BOT (\(+0.034\)); largest are DOS, SSHPATATOR, and WEBATTACK (\(+0.049\)). No label reverses the sign.

**Table II.** Mean ROUGE-1 by true label.

| Attack | \(n\) | No RAG | RAG + SHAP | B − A |
|--------|--:|-------:|-----------:|------:|
| BENIGN | 23 | 0.126 | 0.348 | +0.222 |
| BOT | 23 | 0.108 | 0.288 | +0.180 |
| DDOS | 22 | 0.148 | 0.367 | +0.219 |
| DOS | 22 | 0.160 | 0.381 | +0.221 |
| FTPPATATOR | 22 | 0.149 | 0.314 | +0.165 |
| OTHERS | 22 | 0.110 | 0.334 | +0.224 |
| PORTSCAN | 22 | 0.136 | 0.317 | +0.181 |
| SSHPATATOR | 22 | 0.176 | 0.346 | +0.170 |
| WEBATTACK | 22 | 0.135 | 0.306 | +0.171 |
| Overall | 200 | 0.138 | 0.333 | +0.195 |

Largest ROUGE-1 lift: OTHERS (\(+0.224\)) and BENIGN (\(+0.222\)). Smallest: FTPPATATOR (\(+0.165\)). DoS/DDoS keep the highest B ROUGE-1 (0.381 / 0.367) because those gold concatenations share more rate-limit / scrubbing phrasing with the retrieved parents.

**Table III.** Mean ROUGE-L by true label.

| Attack | \(n\) | No RAG | RAG + SHAP | B − A |
|--------|--:|-------:|-----------:|------:|
| BENIGN | 23 | 0.059 | 0.117 | +0.058 |
| BOT | 23 | 0.061 | 0.119 | +0.058 |
| DDOS | 22 | 0.072 | 0.179 | +0.107 |
| DOS | 22 | 0.075 | 0.186 | +0.111 |
| FTPPATATOR | 22 | 0.075 | 0.126 | +0.050 |
| OTHERS | 22 | 0.064 | 0.142 | +0.078 |
| PORTSCAN | 22 | 0.062 | 0.104 | +0.042 |
| SSHPATATOR | 22 | 0.081 | 0.154 | +0.073 |
| WEBATTACK | 22 | 0.067 | 0.128 | +0.061 |
| Overall | 200 | 0.068 | 0.139 | +0.071 |

ROUGE-L follows ROUGE-1: DDoS/DoS gain the most longest-common-subsequence overlap; PORTSCAN the least (\(+0.042\)), still strictly positive.

---

## IV. What this does and does not show

A positive delta means B’s emitted text (parents + rationale) is closer to the gold policy concat than A’s rationale. It does **not** score whether the chosen catalog action is the gold action. Primary \(\in W[\mathrm{true}]\) is 200/200 for A and 194/200 for B; action tables live in `experiments/rag/hybrid_eval200/report.md`. No Commit / Apply. No McNemar.

Rebuild figures from the CSV (no new LLM calls):

```
cd backend
python scripts/rag_retrieval_scoring.py --offgold-n 200 --offgold-report-only --systems LLM_only,RAG_RANKING
```
