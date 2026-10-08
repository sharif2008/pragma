# Notes: later updates to Pragma_v2.tex (do not edit the paper yet)

`docs/Pragma_v2.tex` is left unchanged. This file is the only place to track what to revise after a clean rerun.

Source run: `experiments/detect-train/` (`model_comparison_20261008_025340.json`, schema 1.2 `agentic_features.json`).
Do **not** paste these numbers into the paper until seed + cloned checkpoint + 88-d NN. Then edit `docs/Pragma_v2.tex` (and the copy in `docs/Pragma_CCNC_Section3A_Detection.tex`) from this checklist.

## Table~\ref{tab:vfl} (lines ~896–910)

| | Acc. | Mac. Rec. | Mac. F1 |
|--|------|-----------|---------|
| Paper VFL | 0.9628 | 0.9861 | 0.8633 |
| This VFL | 0.9597 | 0.9855 | 0.8635 |
| Paper NN (caption: 88 feat.) | 0.9614 | 0.9829 | 0.8617 |
| This NN (`input_dim` **97**) | 0.9572 | 0.9822 | 0.8409 |
| Paper VFL − NN | +0.0014 | +0.0032 | +0.0016 |
| This VFL − NN | +0.0024 | +0.0033 | +0.0226 |

Split sizes match the paper (249 854 / 62 464 / 78 080, seed 42 on the split). VFL **macro-F1 already matches**. Accuracy is ~0.3 points lower. The NN F1 gap is **not** comparable to the paper: this baseline saw duplicated shared columns.

**Later:** replace table + the paragraph that says the F1 difference is “only 0.0016”, after a fair NN rerun. Keep the *interpretation* (locality, not an accuracy win) if the gap stays small.

## What to fix in the paper (wording)

| Location | Paper now | This training / code | Later action |
|----------|-----------|----------------------|--------------|
| Table~\ref{tab:vfl} “strict locality” | implies disjoint columns | 9 shared signals (97 listings / 88 unique) — same paragraph already says this | Drop “strict locality”; say “three-party VFL, 9 shared signals” |
| Table~\ref{tab:vfl} “88 feat.” | unique columns | NN concatenates party slices → **d = 97** | Either train NN on unique 88 (preferred, matches caption) then keep “88 feat.”, or change caption to “concatenated party slices, d=97” |
| Table pipeline, Fusion | \(192 \to 192 \to 128 \to K\) | `ActiveClassifier`: \(192 \to 128 \to K\) (body §Coordinator already correct) | Fix the setup table to \(192 \to 128 \to K\) |
| Feature partitioning | “rule-based classifier on feature names” | `backend/storage/agentic_features.json` lists | Say catalog / JSON assignment |
| Same paragraph | “no party sees another party’s columns” | nine columns appear in two parties | **Done:** “Parties do not exchange raw columns; the catalog may overlap.” |
| §Training | “restoring the best checkpoint” | `state_dict().copy()` shares storage; NN best epoch **95**, test used later weights | Fix code, then the sentence is true |
| §Training | seed 42 | only `train_test_split`; no `torch.manual_seed` | Add seed; mention dropout is seeded |
| §Explainability | KernelSHAP “on the fused 192-d representation” | SHAP is on the **distilled surrogate**, not live VFL | Align with the Attribution paragraph |

## What to fix in code before reprinting the table

1. Clone checkpoint tensors (`v.detach().clone()`), not `state_dict().copy()`.
2. `torch.manual_seed(42)` (+ numpy / cuda) in `detect_train.py`.
3. Centralized NN on **88 unique** columns (dedupe the concat), matching the table caption and \(d\) in the baseline paragraph.
4. Optional: fit scalers on **train** only (paper is silent; current fit is on the full frame).

Do **not** revert schema 1.2 column lists to reprint 0.9628. Access / Perimeter / Endpoint 23/32/42 is what Table~\ref{tab:detect-parties} already describes. The old notebook lists had the same *counts* and different *columns*; that is why Acc. moved.

## Leave as-is (matches this run)

- \(d_p = 23/32/42\), \(K=9\), CICIDS-2017, 88 unique numeric columns
- Local encoder \(d_p \to 128 \to 64\), dropout 0.2 / 0.1
- Adam \(10^{-3}\), wd \(10^{-4}\), clip 1, class-weight clamp \(20\times\), plateau 0.5 / 15, max 100 epochs, val every 5, early-stop patience 20 / \(\Delta=0.001\)
- Centralized MLP \(d \to 256 \to 128 \to 64 \to K\) (architecture only; input width is the issue)
- Distillation 50 epochs, \(T=3.0\), KernelSHAP 100 background / 200 perturbations

## Per-class (this run; paper has no per-class table)

BOT F1 is ~0.33 (VFL) / ~0.31 (NN) with recall \(\approx 1\) — BENIGN false positives. OTHERS has 38 test rows; NN F1 0.69 vs VFL 0.82, which inflates macro-F1 gaps. If a per-class table is added later, footnote support for BOT / OTHERS.
