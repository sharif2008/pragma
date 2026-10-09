# rag_reason_500

500 held-out test flows, **disjoint** from gold cores (`fixtures/gold-100`).

Paper path **C** (`RAG_RANKING`) only: **one LLM call per flow**.

Headline: per-attack mitigation scorecard (action correctness ↑, policy compliance ↑, unsafe action rate ↓, evidence support ↑). These 500 rows are **not** gold-100.

## Class counts (true label)

| class | n | AEVARA catalog |
|-------|--:|---------------:|
| BENIGN | 59 | 2 |
| BOT | 59 | 8 |
| DDOS | 59 | 8 |
| DOS | 59 | 7 |
| FTPPATATOR | 59 | 8 |
| OTHERS | 28 | 7 |
| PORTSCAN | 59 | 8 |
| SSHPATATOR | 59 | 8 |
| WEBATTACK | 59 | 8 |

Do not resample. Do not include any gold `split_index`.
Blockchain / e2e should reuse this fixture.
