# Shared Expert Pools: Sublinear Infrastructure Scaling for Mixture-of-Experts Models

**Aleksei Ketslakh — Technical Architecture Proposal (June 2026)**

---

## Abstract

This document proposes **shared expert pools** — an architectural principle for Mixture-of-Experts (MoE) models in which groups of adjacent layers draw from the same pool of experts instead of maintaining independent per-layer expert sets. Applied to a concrete 1.15T-parameter MoE design (235B active), this approach yields a **10× reduction in effective parameter footprint** relative to a per-layer equivalent of comparable quality (~4.3T), enables **48 logical model replicas on a single 72-GPU server node** at 1.5 GPUs per replica, and produces a **3–10× reduction in inference deployment cost** — all while retaining an estimated 90–97% of the quality of the per-layer counterpart. The principle is architecture-agnostic and applies to existing MoE designs (demonstrated below on a Kimi-K2-like model), with benefits that scale superlinearly as models grow larger.

No experimental validation has been performed. This is an architectural hypothesis supported by parameter-level arithmetic, hardware deployment analysis, and structural arguments for quality preservation.

---

## 1. The Problem: MoE Deployment Economics

Modern MoE models achieve strong quality at moderate active compute cost, but their **deployment** remains expensive. A 1T+ MoE model with per-layer expert sets stores hundreds of billions of expert parameters that are mostly idle for any given token — yet all of them must reside in VRAM.

Consider deploying a hypothetical 4.3T MoE model (235B active) on NVIDIA Rubin NVL72 GPUs (288 GB VRAM each):

- Model weights in FP8: ~4.3 TB → **~16 GPUs** just for weights.
- With KV-cache and activations: ~18–20 GPUs per replica.
- On a 72-GPU node: **~4 replicas**, ~48 concurrent sequences (batch 12).

This is the baseline we aim to beat — **by an order of magnitude**.

---

## 2. Core Principle: Shared Expert Pools

The key insight: **adjacent MoE layers within the same processing block operate on statistically similar hidden-state distributions, and therefore do not require independent expert sets.**

In a standard MoE, each layer `l` has its own pool of `E` experts. In a 96-layer model with 256 experts per layer, this means 96 × 256 = 24,576 independent experts.

With shared pools, we group `G` adjacent layers and assign them a single shared pool. For `G = 4`, the same 96-layer model needs only 24 pools × 256 experts = 6,144 experts — a **4× reduction** in expert parameters, with every layer still selecting from 256 candidates.

This is not a new concept in kind — **Grouped Query Attention (GQA)** applies the same principle to KV-heads, sharing key-value projections across query heads. Shared expert pools extend the sharing principle from attention to feed-forward experts, at a higher level of the hierarchy and with correspondingly larger savings.

---

## 3. Reference Architecture

To ground the discussion, we present a concrete 1.15T model designed around shared pools.

### 3.1 Model Structure

**Layer pattern:** 4 dense → [8 MoE → 1 dense] × 12

| Component | Count |
|---|---|
| Dense layers | 16 (4 prefix + 12 block-terminal) |
| MoE layers | 96 |
| Shared expert pools | 24 (1 pool per 4 adjacent MoE layers) |
| Total layers | 112 |

### 3.2 Dimensions

| Parameter | Value |
|---|---|
| d_model | 12,288 |
| Attention heads (Q / KV) | 96 / 12 (GQA 8:1) |
| Head dimension | 128 |
| Vocabulary | 458,752 tokens (tied embeddings) |

### 3.3 Feed-Forward Design

**Dense-layer FFN** (SwiGLU): d → 3.75d → d = **11.25d² params**, total layer size **13.5d²**

**MoE-layer components:**
- Micro dense-FFN (SwiGLU): d → 0.75d → d = **2.25d² (~340M params)** — always active, layer-specific
- MLP Router: d → 1024 → 256 = **12.85M params** — layer-specific
- Layer-specific total: **4.5d² + 12.85M**

**Shared pool** (per pool of 4 layers):
- 256 experts, each SwiGLU: d → 0.375d → d = **1.125d² (~170M params)**
- Pool total: **288d²**
- All 24 pools: **6,912d²**

**Routing:** top-8 out of 256 experts (3.1% sparsity)

### 3.4 Critical Property: Uniform Active Layer Size

Active compute per MoE layer = micro FFN (2.25d²) + attention (2.25d²) + 8 experts (8 × 1.125d² = 9d²) = **13.5d²**

Dense layer = attention (2.25d²) + FFN (11.25d²) = **13.5d²**

**Every layer in the model, whether dense or MoE, has identical active compute cost.** This is not a coincidence — it is a design constraint that yields uniform latency per layer, trivial pipeline scheduling, and predictable resource allocation.

### 3.5 Model Size

| Metric | Value |
|---|---|
| Nominal parameters | **~1.15T** |
| Active parameters per token | **~235B** |
| Per-layer equivalent (if pools were per-layer) | **~4.28T** |

---

## 4. Quality Preservation: Four Mechanisms

The central concern with shared pools is inter-layer interference — experts receiving conflicting gradient signals from different layers. Four architectural mechanisms work in concert to mitigate this.

### 4.1 Layer-Specific Compute (Micro Dense-FFN)

Each MoE layer retains a dedicated 340M-parameter SwiGLU FFN that is **not shared**. This constitutes 1/3 of the layer's active compute (4.5d² out of 13.5d²). Even if expert routing returns suboptimal results, the layer has guaranteed layer-specific processing capacity. This also creates a clean gradient highway — the micro FFN receives gradients without cross-layer interference, stabilizing early-layer training.

### 4.2 Native Shared Training

Experts are trained in shared pools **from the very beginning** — they never learn to be layer-specific and then painfully adapt to sharing. From initialization, each expert knows it will be called from different depths. This is fundamentally different from "train per-layer, then merge pools," which induces catastrophic forgetting.

### 4.3 Expert Capacity (170M Parameters)

At 170M parameters, each expert is a mini-network with sufficient internal capacity for **conditional behavior across layers**. A small expert (e.g., 44M as in Kimi-K2) can only learn one function applied uniformly regardless of calling depth. A 170M expert can develop internal gating that modulates its behavior based on input distribution — which implicitly encodes depth information via the hidden-state statistics.

### 4.4 Phase Proximity Within Blocks

This is the foundational argument. The block structure [8 MoE → 1 dense] creates a **hierarchical processing rhythm**:

- Within a block, 8 MoE layers apply incremental residual updates to the same representation. The hidden state evolves slowly — each layer adds a small correction, not a phase shift.
- The dense layer at the end of each block acts as a **synchronizer** — it sees all features without routing, applies a full-width FFN, and produces the "report" for the next block. This is where the real phase transition occurs.
- Each shared pool covers 4 adjacent MoE layers **within the same block** (2 pools per block). Pool boundaries never cross block boundaries.

Quantitatively: if the dense layer absorbs ~30–40% of each block's representational shift (it is the only full-width, unrouted computation), then the 4 MoE layers in one pool cover only **~2.5–3% of the model's total processing path** — even less than the naive estimate of 4/112 = 3.6%.

At this level of phase proximity, the hidden-state distributions seen by an expert called from layer 2 vs. layer 4 of the same pool are **statistically close**. Inter-layer interference is not suppressed by force — it is **architecturally minimal**.

### 4.5 Expected Quality Retention

Given all four mechanisms, we estimate **90–97% quality retention** relative to a per-layer 4.28T model. Even at the pessimistic bound of 80%, the model would still outperform a conventional 1.15T per-layer MoE — because every layer selects from 256 experts (fine-grained specialization) rather than the ~64 experts affordable in a per-layer 1.15T budget (coarse specialization). Selection quality × expert quality: the first factor grows faster than the second degrades.

---

## 5. Hardware Deployment: Trunk-Expert Disaggregation

Shared pools enable a deployment architecture that **separates stateful and stateless model components** onto different GPUs.

### 5.1 Two GPU Types

**Trunk-GPU** — carries the "model body":
- Dense + MoE layers (attention, micro FFN, routers) in FP8: ~99 GB
- Vocabulary embeddings: ~5.6 GB
- KV-cache for batch of 12 sequences (32k context, int8, +25% overhead): ~168 GB
- Total: ~273 GB of 288 GB (15 GB reserve)

**Exp-GPU** — carries shared expert pools:
- 6 pools per GPU (4 GPU subtypes by depth)
- ~261 GB of 288 GB utilized (~27 GB reserve)

### 5.2 Node Layout (NVL72)

| GPU Type | Count | Role |
|---|---|
| Trunk-GPU | 48 | Full model body + KV-cache, each is a logical replica |
| Exp-GPU | 24 | Shared expert pools (6 per subtype × 4 subtypes) |
| **Total** | **72** | |

### 5.3 The Key Metric

Each trunk-GPU is a **self-contained inference unit** except for expert lookups. The 24 exp-GPUs are shared across all 48 trunks via any-to-any routing over NVLink (3.6 TB/s on NVL72).

**Effective cost per logical replica: 1 trunk + 24/48 exp = 1.5 GPU.**

- 48 replicas × batch 12 = **576 concurrent sequences** from a single NVL72 node.
- Conventional dense 1.15T (5–6 GPU/replica): ~12 replicas → ~144 sequences.
- Conventional per-layer 4.28T (16 GPU/replica): ~4 replicas → ~48 sequences.

**The shared-pool architecture serves 4× more concurrent sequences than a conventional 1.15T deployment, and 12× more than the 4.28T per-layer equivalent — from the same hardware.**

### 5.4 Throughput Estimate

With Rubin at ~30 PFLOPS FP8 and active size 235B:
- Ideal throughput per replica: ~95,744 tokens/sec
- At 30–50% compute efficiency: **28,700–47,800 tokens/sec per replica**
- Full-batch (12 sequences × 32k context) generation: **~10 seconds**

Latency is slightly higher than conventional (single-replica) deployment. But modern reasoning models routinely take 10–30+ seconds for thinking. The trade-off — modest latency increase for 4–12× throughput — is overwhelmingly favorable.

---

## 6. Universality: Application to Existing Architectures

Shared expert pools are not specific to the reference architecture above. The principle applies to **any MoE model with per-layer expert sets**.

### 6.1 Example: Kimi-K2-like Architecture

Kimi-K2: 1.04T nominal, 32B active, 384 experts × 44M each × 60 layers.

Applying conservative sharing (×2, grouping every 2 adjacent layers):
- Expert parameters reduced from ~1.01T to ~505B
- New nominal size: **~560B** (same 32B active)
- Same quality characteristics — every layer still selects from 384 experts

Hardware deployment at 4:1 trunk:exp ratio:
- ~1.25 GPU per logical replica
- Model body on trunk: ~11–15 GB (leaving ~273 GB for massive KV-cache batching)
- **3.2× deployment savings** vs. conventional

### 6.2 Comparison: Why Shared Pools Unlock Superior Configurations

The reference architecture simultaneously achieves wider, deeper, and fatter than Kimi-K2 — at comparable nominal parameter count:

| Property | Reference (1.15T) | Kimi-K2 (1.04T) | Ratio |
|---|---|---|---|
| d_model | 12,288 | 7,168 | 1.71× |
| Total layers | 112 | 61 | 1.84× |
| Expert size | 170M | 44M | 3.9× |
| Per-token guaranteed FFN | 340M (micro FFN) | 44M (shared expert) | 7.7× |
| Dense synchronizers | 12 (every 9 layers) | 1 (first layer only) | 12× |

In conventional MoE, these three axes (width, depth, expert capacity) compete for the same parameter budget. Shared pools **break this trade-off** by compressing expert storage 4×, freeing budget for all three axes simultaneously.

---

## 7. Superlinear Scaling of Architectural Advantage

A counterintuitive property: the benefit of shared pools **increases** with model scale.

Three mechanisms drive this:

1. **Depth:** More layers → each layer is a smaller fraction of total path → phase proximity between adjacent layers increases → sharing becomes safer → more aggressive sharing ratios are possible.

2. **Expert capacity:** Larger d_model → fatter experts → more internal capacity for cross-layer conditioning → less interference per unit of sharing.

3. **Width:** Larger d_model → GQA degrades less → more aggressive GQA ratios are possible → savings in attention budget flow into expert capacity → fatter experts → safer sharing. (Positive feedback loop.)

This produces a **quality multiplier κ** (ratio of effective quality to nominal parameter count) that grows with scale:

| Nominal Size | Sharing Ratio | Estimated κ | Effective Quality Equivalent |
|---|---|---|---|
| 200–400B | ×2–3 | ~2× | 400–600B |
| 600B–1T | ×3–4 | ~3–4× | 1.8–2.4T |
| 1–2T | ×4–6 | ~4–5× | 4–10T |
| 2T+ | ×5–6 | ~5–6× | 10T+ (unverifiable) |

This inverts the standard scaling dynamic: conventional scaling yields diminishing returns per dollar, while **architectural efficiency via shared pools yields increasing returns per dollar** as models grow. Beyond a crossover point, the gap widens monotonically.

At the largest scales (2T+ nominal → 10T+ effective), empirical verification becomes impossible — building a native 10T+ per-layer model to serve as a baseline would cost more than the shared-pool version saves. The architecture's quality claim becomes unfalsifiable from above, verifiable only by dominating all existing benchmarks.

---

## 8. Operational Modularity

The trunk/exp separation creates an operationally flexible deployment with **two independent scaling levers**.

### 8.1 Independent Scaling

- **Need more throughput?** Add trunk-GPUs. Each new trunk = one new logical replica.
- **Need more expert capacity per replica?** Add exp-GPUs. Reduces slot contention, lowers latency.

The scaling unit is a **single GPU**, not a multi-GPU replica. Granularity is maximally fine.

### 8.2 Discrete Configuration Modes

The number of "slots" (concurrent replicas served) per exp-GPU must divide evenly into the number of pools it hosts. This creates a small, well-defined set of operating modes:

**Reference architecture (6 pools per exp-GPU):**
- 2 slots (3 pools per slot) — aggressive, maximum throughput
- 3 slots (2 pools per slot) — conservative, higher per-replica capacity

**Kimi-like (15 pools per exp-GPU on Rubin):**
- 3 slots (5 pools per slot)
- 5 slots (3 pools per slot)

Operators choose from 2–3 presets, not a continuous optimization space. Like fixed gear ratios — simpler, more reliable, sufficient flexibility.

### 8.3 Adaptive Operation

Start aggressive (max slots, max throughput). If exp-GPUs bottleneck under load, add more exp-GPUs or reduce slots. No model retraining, no architecture changes — only routing configuration.

### 8.4 Cross-Hardware Portability

The same model deploys on different GPU generations by adjusting pool-per-GPU counts:

| GPU | VRAM | Pools per exp-GPU | Viable slots |
|---|---|---|---|
| Rubin (288 GB) | 288 GB | 6 | 2, 3 |
| H200 (141 GB) | 141 GB | 6 (tighter) | 2, 3 |
| H100 (80 GB) | 80 GB | 3 | 1, 3 |

The model is not tied to a specific hardware generation. Upgrading GPUs changes the deployment config, not the model.

---

## 9. Economic Impact

### 9.1 Per-Node Savings

On a single NVL72 node (estimated cost $3–5M for Rubin generation):

| Deployment | Replicas | Concurrent Seqs | Cost/Replica |
|---|---|---|---|
| Shared pools 1.15T | 48 | 576 | 1.5 GPU |
| Conventional 1.15T | 12 | 144 | 5–6 GPU |
| Per-layer 4.28T | 4 | 48 | 16+ GPU |

### 9.2 Fleet-Scale Savings

At 100 NVL72 nodes (~$400M infrastructure):
- Shared pools: 4,800 replicas, 57,600 concurrent sequences
- To match this throughput with conventional 1.15T: ~400 nodes (~$1.6B)
- To match with per-layer 4.28T: ~1,200 nodes (~$4.8B)

**Savings: $1.2–4.4B in avoided infrastructure cost.** Ongoing savings in power, cooling, and operations compound over the deployment lifetime.

### 9.3 Cost per Token

With 4× higher throughput per node at comparable power draw, cost per million tokens drops proportionally. For a provider currently charging $15/M input tokens, the architectural margin enables either a ~4× price reduction (competitive advantage) or a ~4× profit increase (business advantage).

### 9.4 Democratization

Companies with training budgets for 200B models can now target quality levels previously requiring 400–600B — entering the competitive tier of much larger labs. This shifts frontier AI capability from a pure capital-scaling game to an **architecture-scaling game**, broadening the competitive landscape.

---

## 10. Open Questions

1. **Training recipe.** No public precedent exists for training shared expert pools at the 1T+ scale. Specific challenges include auxiliary loss design for cross-layer load balancing, learning rate scheduling for shared vs. layer-specific parameters, and potential curriculum strategies (progressive sharing ratios during training). These are engineering problems, not fundamental obstacles — analogous to the challenges GQA posed before becoming standard.

2. **Empirical quality validation.** The 90–97% quality retention estimate is based on structural arguments (phase proximity, expert capacity, protection mechanisms). Controlled experiments at smaller scales (7B–70B) would establish the actual κ curve and validate or refine the theoretical bounds.

3. **Expert routing dynamics.** With an MLP router (d → 1024 → 256, 12.85M params) instead of a linear router, per-layer routing latency increases. The wall-clock cost relative to a linear router needs profiling at scale. The richer router may also enable layer-aware routing strategies that further reduce interference.

4. **Cold retrieval mechanism.** The 2M-token extended context uses LongRoPE + YaRN + RAG-attention for retrieval from compressed cold storage. The specific retrieval architecture (Infini-attention, kNN-attention, or custom) affects long-range reliability and requires empirical tuning.

5. **Scaling law for κ.** The superlinear growth of the quality multiplier is currently a structural prediction. Establishing an empirical scaling law (κ as a function of model size, sharing ratio, and expert capacity) would enable precise architecture selection for target quality levels.

---

## 11. Conclusion

Shared expert pools represent a **deployment paradigm shift** for MoE models:

- **1.5 GPUs per replica** of a 1.15T model (vs. 5–6 conventional, 16+ for the per-layer equivalent)
- **48 replicas on a 72-GPU node** (vs. 12 conventional, 4 per-layer)
- **90–97% quality** of a model 4× larger in nominal parameters
- **3–10× cost reduction** in inference infrastructure, scaling superlinearly with model size
- **Operationally modular** deployment with independent scaling levers and cross-hardware portability

The principle is not architecture-specific. It applies to any MoE model with per-layer expert sets, with conservative sharing (×2) yielding significant savings even without architectural co-design.

The required investment is **one-time**: developing and validating a training recipe for shared pools. The return is **perpetual and multiplicative**: every model trained with this principle, deployed on every node, serving every request, benefits from sublinear infrastructure scaling.

The question is not whether this trade-off is favorable. The question is who captures it first.

---

*No experimental validation. All estimates are theoretical, derived from parameter-count arithmetic and structural analysis. Empirical verification at scale is the necessary next step.*
