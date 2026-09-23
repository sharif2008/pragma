# III-A. Detection Layer (Detect)

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

---

**Feature partitioning (controlled proxy, not three sensors).** Identifiers, addresses, and timestamps are dropped. The remaining 88 CICFlowMeter numeric columns from CICIDS-2017 are assigned to three evidence groups by a rule-based classifier on feature names:

| Party \(p\) | Enterprise label | \(d_p\) | Evidence | Typical attacks |
|-------------|------------------|---------|----------|-----------------|
| 1 | Access / ISP | 23 | Volume / rate (durations, packet and byte totals, aggregate SYN/UDP/TCP counters) | DoS / DDoS |
| 2 | Perimeter / IDS | 32 | Packet-shape and ports (length statistics, unique/vulnerable/HTTP/DNS ports, src\(\rightarrow\)dst flags) | Scan, web |
| 3 | Endpoint / EDR | 42 | Timing / direction (IAT, bidirectional and reverse-path TCP flags) | Brute force, host-facing |

The catalog lists 97 assignments over 88 unique columns; nine signals are shared so one group can corroborate another without a raw-feature exchange. This is a statistical partition of a **single** flow-level feature set, not three independently instrumented systems. We present it under Access / Perimeter / Endpoint because those are the enterprise layers where each signal type would originate. Original CICIDS labels are collapsed to \(K=9\): BENIGN, DDOS, DOS, SSHPATATOR, FTPPATATOR, PORTSCAN, WEBATTACK, BOT, and OTHERS (classes with fewer than 200 flows, including Infiltration and Heartbleed). Features at each party are standardized with a party-local `StandardScaler` and stored as 32-bit tensors. The aligned index set is split 64\% / 16\% / 20\% train / validation / test by stratified sampling (seed 42), so every party sees the same flows and no party sees another party's columns.

---

**Local MLP encoder (per party).** Each party \(p\in\{1,2,3\}\) owns an encoder \(f_p\) that never leaves that party. It is a two-layer MLP, not a linear projection:

\[
x_i^p \;\in\mathbb{R}^{d_p}
\;\xrightarrow{\;W_1\in\mathbb{R}^{128\times d_p}\;}
\mathbb{R}^{128}
\;\xrightarrow{\mathrm{ReLU},\;\mathrm{Dropout}(0.2)}\;
\;\xrightarrow{\;W_2\in\mathbb{R}^{64\times 128}\;}
\mathbb{R}^{64}
\;\xrightarrow{\mathrm{ReLU},\;\mathrm{Dropout}(0.1)}\;
h_i^p.
\]

Only \(h_i^p\) is transmitted. Three such encoders run in parallel with independent weights \(\theta_p\). Concatenation---not summation---is used at the coordinator so each 64-dimensional block remains identifiable for SHAP.

---

**Coordinator MLP (VFL classifier).** The coordinator concatenates

\[
h_i = [h_i^1 \,\|\, h_i^2 \,\|\, h_i^3] \in \mathbb{R}^{192}
\]

and applies a shared active classifier \(g\):

\[
h_i
\;\xrightarrow{\;192\rightarrow 128,\;\mathrm{ReLU},\;\mathrm{Dropout}(0.2)\;}
\;\xrightarrow{\;128\rightarrow K\;}
z_i,\qquad
\hat{y}_i=\arg\max_k z_{i,k}.
\]

Logits \(z_i\) are unnormalized; softmax lives only in the loss and in the confidence \(\mathrm{conf}_i=\max_k\mathrm{softmax}(z_i)_k\) that Reason will consume. The end-to-end VFL model is therefore three local MLPs plus one coordinator MLP, trained by back-propagating through the concatenation. Gradients of \(g\) return to each \(f_p\); raw \(x_i^p\) still never move.

---

**Training (shared recipe).** Both VFL and the centralized baseline use the same optimization so the comparison is not a training artifact. The loss is class-weighted cross-entropy. Inverse-frequency weights are clamped at \(20\times\) so rare classes (BOT, WEBATTACK, OTHERS) cannot dominate the gradient. The optimizer is Adam (\(\mathrm{lr}=10^{-3}\), weight decay \(10^{-4}\)); gradients are clipped at \(\ell_2=1\). A plateau scheduler halves the learning rate when validation loss stalls (factor \(0.5\), patience 15, floor \(10^{-6}\)). Training runs at most 100 epochs, validates every 5 epochs, and early-stops on macro-\(F_1\) (patience 20, \(\Delta_{\min}=0.001\)), restoring the best checkpoint. Reported metrics are accuracy, macro-recall, and macro-\(F_1\), plus a per-class report against the centralized model.

---

**Centralized all-feature baseline (comparison only).** The privacy-preserving claim is empty unless VFL is compared with a model that **is** allowed to see every column. We therefore concatenate the three already-scaled slices into one vector \(\tilde{x}_i=[x_i^1\|x_i^2\|x_i^3]\in\mathbb{R}^{d}\) (\(d=97\) listings / 88 unique columns) and train a non-federated MLP of comparable depth:

\[
\tilde{x}_i
\;\rightarrow\; 256 \;\xrightarrow{\mathrm{ReLU},\;\mathrm{Dropout}(0.2)}\;
128 \;\xrightarrow{\mathrm{ReLU},\;\mathrm{Dropout}(0.2)}\;
64 \;\xrightarrow{\mathrm{ReLU},\;\mathrm{Dropout}(0.2)}\;
K.
\]

It uses the **same** split, class weights, Adam settings, scheduler, gradient clip, and early-stopping rule. It is deliberately not a shallow network: a weaker centralized model would make VFL look better for the wrong reason. The centralized MLP is an evaluation instrument. It is not deployed, it does not produce SHAP for planning, and it does not feed Reason. The live path is VFL embeddings only.

---

**Attribution via distilled surrogate.** KernelSHAP on the live VFL stack would require hundreds of forward passes per explained flow, including three local encoders. That is impractical in a detection pipeline. We therefore freeze the trained VFL model as teacher, form the 192-d concatenation of its embeddings, and train a compact surrogate \(m\) (\(192\rightarrow 128\rightarrow 64\rightarrow K\), dropout \(0.2\) on the first hidden layer) by soft-label distillation: KL divergence between temperature-smoothed distributions at \(T=3.0\), scaled by \(T^2\), Adam as above, 50 epochs. Teacher--student agreement (hard-label match, KL, logit MSE) is measured on the test set before any SHAP value is trusted.

KernelSHAP then explains \(m\), not the three encoders: background size 100, perturbation budget 200 per instance, up to 200 test flows. Values lie in \(\mathbb{R}^{192}\) and are **summed inside each 64-d block** to one score per evidence group. The group with the largest mean \(\lvert\phi\rvert\) is \(p_i^{\star}\in\{\)Access / ISP, Perimeter / IDS, Endpoint / EDR\(\}\). Top-3 features per block and top-5 features overall are retained; those are the only feature names Section III-B is allowed to put in the retrieval query and the prompt.

---

**Handoff to Reasoning.** The Detection Layer emits, for each flow \(i\),

\[
(\hat{y}_i,\; \mathrm{conf}_i,\; p_i^{\star},\; \text{three domain shares},\; \text{top features by }\lvert\phi\rvert).
\]

This closes the detection-without-centralization problem: Access, Perimeter, and Endpoint never pooled raw columns, and the coordinator still produced a class competitive with the all-feature MLP. The output is still **evidence, not a plan**. Converting that record into a catalog action with a `network_tier`---without inventing names or ignoring \(p_i^{\star}\)---is the failure the Reasoning and Planning Layer exists to close.
