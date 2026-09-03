# Shared Expert Pools: An Architectural Framework for Efficient Mixture-of-Experts Design and Deployment

**Aleksei Ketslakh — Architectural Framework Proposal (August 2026)**

---

## Abstract

This document presents **shared expert pools** — a set of architectural principles for Mixture-of-Experts (MoE) models in which groups of adjacent layers draw from the same pool of experts instead of maintaining independent per-layer expert sets.

Unlike prior approaches to expert sharing (e.g., UniPool) that introduce layer-conditioning mechanisms to help shared experts adapt to different depths, this framework takes the opposite approach: it ensures that **experts do not need layer-specific adaptation at all**. A sufficiently large layer-specific micro dense-FFN absorbs all depth-dependent processing before experts are invoked, leaving only generic refinement work for the shared pool — work that is inherently portable across layers.

The framework defines:
- **Structural principles** for MoE layer design (sequential micro dense-FFN → routed experts, uniform active compute constraint)
- **A training recipe** (phased dense/MoE training, two-stage router learning)
- **Deployment architecture** (logical replicas, trunk/exp GPU disaggregation, conveyor pipeline)
- **A quality estimation metric** (C-factor) that combines observable pool breadth with empirically calibrated sharing efficiency

A reference 132B-nominal model is presented as one instantiation of these principles, deployable on 8×H100 GPUs with 6 logical replicas (4 MoE + 2 dense-like) at 1–1.5 GPU per replica.

**No experimental validation has been performed.** This is an architectural hypothesis. All structural principles are presented as testable claims; all specific numerical parameters are presented as reasonable starting points for empirical search, not as final values.

---

## 1. The Problem: MoE Deployment Economics

Modern MoE models achieve strong quality at moderate active compute cost, but their **deployment** remains expensive. A typical MoE model stores hundreds of billions of expert parameters — 80–90% of its total size — that are mostly idle for any given token, yet all of them must reside in VRAM for any replica to function.

This creates a fundamental inefficiency: the unit of scaling is a full model replica (all weights duplicated), but the unit of useful work is a single forward pass that touches only a small fraction of those weights. Adding throughput means duplicating the entire model, including the 80–90% of parameters that the new replica will also leave mostly idle.

The core question this framework addresses: **can we separate the part of the model that must be per-replica from the part that can be shared across replicas — and can we design the architecture so that this separation is not a compromise, but a natural consequence of how the model processes information?**

### 1.1 Related Work and Positioning

This framework intersects several active lines of MoE research but combines them differently.

- **Layer-local shared experts.** DeepSeekMoE [2] introduces always-active shared experts alongside routed experts to capture common knowledge and reduce redundancy. The present framework differs structurally: the micro dense-FFN is layer-specific, expanding, and executed **sequentially before** routing — not as a parallel shared expert branch.

- **Inter-layer expert reuse.** UniPool [1] replaces per-layer expert ownership with a globally shared pool accessed by independent layer routers. Recent tied-expert work [3] also studies reusing experts across nearby layers. The present framework uses local pools shared by adjacent layers and places the main layer-specific transformation before the shared pool, making expert functions more portable by construction rather than by conditioning.

- **Disaggregated MoE serving.** MegaScale-Infer [4], FinDEP [5], and ExpertPlex [6] separate attention-side and expert-side computation with specialized scheduling. The deployment architecture proposed here belongs to the same systems family, but co-designs the model itself around depth-indexed shared pools, logical trunk replicas, and virtual expert slots.

- **Expert redundancy and compression.** Expert pruning and merging methods [7] demonstrate that trained MoE models contain removable or consolidatable expert capacity. This motivates testing whether such redundancy can be avoided architecturally rather than removed post-training.

The contribution claimed here is not merely "experts can be shared." It is the combined hypothesis that **layer-local dense completeness, inter-layer sparse refinement, force-then-relax routing, and trunk/exp deployment form one coherent architecture.**

---

## 2. Architectural Principles

This section presents the structural principles of the framework. Each subsection distinguishes between:
- **Principles** — architectural claims that are either correct or not, independent of specific numbers
- **Parameter space** — ranges of values that appear reasonable but require empirical validation

### 2.1 MoE Layer Anatomy: Sequential Processing

**Principle.** Each MoE layer processes the residual stream in a strict sequence:

```
Attention → Micro Dense-FFN → [MLP Router → Top-k Experts from Shared Pool]
```

This is not a parallel arrangement. The micro dense-FFN processes the residual stream **first**, performing full layer-specific nonlinear transformation. Only then — and only in MoE operating mode — does the router examine the already-processed stream and select experts for additional refinement.

The computational graph can be summarized as:

```
a_l = x_l + Attn_l(Norm_A(x_l))                          # attention
m_l = a_l + MicroFFN_l(Norm_M(a_l))                       # layer-specific dense
S_l = TopK(Router_l(Norm_E(m_l)), k)                      # routing decision
x_{l+1} = m_l + α_l * Σ_{e∈S_l} g_{l,e} * Expert_{p(l),e}(Norm_E(m_l))  # expert refinement
```

where `p(l)` identifies the shared pool assigned to layer `l`, `g_{l,e}` are softmax-normalized routing weights over the selected set `S_l`, and `α_l` is an optional residual-scaling coefficient. In dense-like mode, the expert branch is skipped: `x_{l+1} = m_l`.

The essential property is the **sequential dependency**: both router and experts operate on the output of the layer-specific micro dense-FFN.

This sequential design has two critical consequences:

1. **The router sees layer-specific signal.** Because the micro dense-FFN has already applied its layer-specific transformation, the router's input implicitly encodes depth information. Expert selection is therefore informed by layer context without any explicit layer-conditioning mechanism.

2. **Experts receive pre-processed input.** The work remaining for experts is not "process this raw residual stream at depth L" but rather "refine this already-transformed representation." This residual refinement is generic — analogous to tightening bolts or polishing a surface, operations that do not depend on which stage of the assembly line produced the part.

This is what makes shared pools viable: **experts do not need to be layer-specific because they are never asked to do layer-specific work.**

**Dual operating mode.** The sequential design naturally supports two regimes:
- **Dense-like regime** — processing ends after the micro dense-FFN (top-k = 0). No expert lookup, no communication with exp-GPUs.
- **MoE regime** — after micro dense-FFN, the router selects top-k experts from the shared pool for additional refinement.

Both regimes use the same weights on the same hardware. The difference is purely in whether the expert refinement step is activated.

### 2.2 Micro Dense-FFN: The Layer-Specific Workhorse

**Principle.** The micro dense-FFN is not a "shared expert" in the DeepSeek sense — a small always-on expert running alongside routed ones. It is a **dense-type FFN with expanding intermediate dimension** (inter > 1×d_model), providing full-width nonlinear processing of the residual stream.

The name "micro" refers to it being narrower than the full dense-layer FFN — not to it being a minor component. In this framework it is a primary layer component.

Key properties:

1. **Expanding intermediate dimension.** Unlike expert FFNs (which have contracting intermediates, typically 0.3–0.5×d), the micro dense-FFN has an intermediate dimension larger than d_model. This makes it a genuine dense-layer-class processor, not a miniature expert. The expansion ratio is smaller than in full dense layers, but the processing is qualitatively the same: project up, apply nonlinearity, project down.

2. **Layer-specific ownership.** Each MoE layer has its own micro dense-FFN with independent weights. This is what provides layer identity to the architecture — the micro dense-FFN is the component that "knows" it is at a particular depth. Shared experts do not need this knowledge because their job begins after it has already been applied.

3. **Significant compute share.** The micro dense-FFN accounts for a substantial fraction of the MoE layer's active FFN compute — not a minor addition, but a primary component. If the micro dense-FFN were small (as in DeepSeek's shared expert approach), it could not absorb enough layer-specific work, and experts would be forced to compensate — making sharing problematic.

4. **Active on 100% of training tokens.** Unlike routed experts (which see only the tokens routed to them), the micro dense-FFN processes every token at every layer. This ensures robust layer-specific representations regardless of routing decisions.

**Parameter space.** The micro dense-FFN's intermediate dimension is expected to fall in the range of 1.25–2.0×d, corresponding to roughly 30–50% of the MoE layer's total active FFN compute. The lower bound (1.25×d) may prove sufficient for layer-specific processing; the upper bound (2.0×d, which in SwiGLU is roughly equivalent to 3.0×d in GeLU — a standard value) provides generous capacity. The actual sweet spot is an empirical question.

### 2.3 The FFN Constraint: Uniform Active Compute

**Principle.** In the effective MoE regime, the total active FFN width of every layer in the model — whether dense or MoE — should be uniform:

```
inter_micro + k × inter_exp = inter_dense
```

Where:
- `inter_micro` — intermediate dimension of the micro dense-FFN
- `inter_exp` — intermediate dimension of each expert FFN
- `k` — top-k in the effective regime
- `inter_dense` — intermediate dimension of full dense-layer FFN

This constraint means that **every layer performs the same nominal amount of active FFN computation**, regardless of whether it is a dense layer or an MoE layer. The MoE layer simply splits this budget between two components: layer-specific processing (micro dense-FFN) and generic refinement (routed experts).

**Important caveat.** The constraint equalizes nominal FFN arithmetic and parameter access, not wall-clock latency. A routed MoE layer still includes router execution, dispatch, inter-GPU communication, grouped GEMMs, and result merging. Equal arithmetic width does not guarantee equal runtime. The constraint is a first-order balancing target that simplifies hardware co-design and narrows the search space; actual stage-time parity must be profiled.

This is a **design constraint, not an observation**. It is imposed deliberately to achieve:
- Comparable nominal FFN work across dense and MoE layers
- More predictable pipeline balancing
- Clean separation between "how much compute" (fixed) and "what kind of compute" (dense vs. split)

**Consequences for hyperparameter selection.** Given `inter_exp`, effective-regime top-k is computed, not tuned:

```
k_effective = (inter_dense - inter_micro) / inter_exp
```

The result must be an integer. Capacity-mode top-k is a separate deployment choice and may intentionally exceed this value.

| inter_micro | top-k | inter_dense (result) |
|---|---|---|
| 1.25d | 5 | 3.25d |
| 1.5d | 5 | 3.5d |
| 1.5d | 6 | 3.75d |
| 1.75d | 5 | 3.75d |
| 2.0d | 4 | 3.5d |
| 2.0d | 6 | 4.25d |

*(Assuming inter_exp = 0.4d for illustration. Actual values may vary.)*

Several observations:

1. **Effective-regime top-k is constrained, not independently free.** Once inter_micro and inter_exp are fixed, top-k follows from the equation.
2. **Dense layers are wider than typical.** The constraint produces inter_dense values of 3.25–4.25×d (vs. the more common 2.67–3.5×d). This is a necessary consequence of keeping active compute uniform while giving micro dense-FFN a meaningful expansion ratio.
3. **Top-k remains moderate** (4–8). High top-k would force inter_micro below 1×d, eliminating the expanding nature of the micro dense-FFN and undermining the entire framework.

**Parameter space.** The constraint is the principle; the specific point within it is empirical. The viable region appears to be inter_dense ∈ [3.25d, 4.25d], inter_micro ∈ [1.25d, 2.0d], top-k ∈ [4, 8], with the micro dense-FFN's share of active FFN at 30–50%. Different model sizes and deployment targets may favor different points within this region.

### 2.4 Training Recipe

**Principle.** Training proceeds in coordinated phases designed to ensure that (a) micro dense-FFN learns robust standalone processing, (b) experts learn generic refinement rather than layer-specific functions, (c) routers explore the full shared pool before specializing, and (d) dense-like capability is actively maintained throughout training.

#### Router Training: Force-then-Relax

Router training uses a two-phase auxiliary loss schedule:

1. **Phase 1 — Full pool utilization.** Routers are trained with a strong auxiliary loss that penalizes underutilization of the expert pool. The goal is to force every router to explore and learn to use **all** available experts, preventing premature collapse onto a small subset.

2. **Phase 2 — Relaxed utilization.** The auxiliary loss is gradually reduced, allowing each router to organically drop a small fraction of experts that are genuinely unhelpful for its layer. The expected outcome is that each layer utilizes 80–95% of the shared pool, forming **heavily overlapping layer-specific sub-pools**.

This is critical for the shared pool to function as intended. If routers collapsed onto small, disjoint subsets, the pool would be effectively partitioned — reverting to per-layer experts with extra overhead. The force-then-relax schedule ensures that the pool remains genuinely shared, with each layer maintaining broad access while developing mild specialization.

**Monitoring pool health.** Utilization should be tracked through multiple diagnostics:
- **Thresholded coverage** — fraction of experts receiving meaningful traffic above a stated threshold per layer over a measurement window
- **Entropy-effective expert count** — `N_eff = exp(H(p_l))`, where `H` is the routing entropy; discounts experts that are technically used but receive negligible mass
- **Pairwise overlap** — Jaccard overlap between thresholded expert sets of sharing layers; very low overlap signals hidden repartitioning
- **Load variance and tail load** — to detect device hotspots and expert collapse

The target utilization range (80–95%) is a structural expectation. Utilization below 80% suggests the shared pool is too large or the micro dense-FFN is insufficient; utilization above 95% with no layer differentiation suggests the pool is too small or the router is too weak.

#### Model Training: Three Sequential Stages

Training proceeds through three strictly sequential stages — each builds on the capabilities established by the previous one:

1. **Stage 1 — Dense-like warm-up** (~10–30% of training). Only attention and micro dense-FFN are active; routers and experts are frozen or absent. The micro dense-FFN learns to perform **autonomous, complete processing** of the residual stream — producing a "rough but complete" output at every layer. This establishes a quality floor: even without experts, every layer does meaningful work.

2. **Stage 2 — MoE regime** (~60–80% of training). Routers and experts are activated. The micro dense-FFN continues its work; experts learn to add precision on top of it. A fraction of batches should continue to run in dense-like mode to prevent the micro dense-FFN from becoming dependent on experts. Since the majority of model parameters reside in the expert pools, this stage requires the largest share of training compute. The micro dense-FFN does not "unlearn" its standalone capability — it continues to do its full job, while experts provide additional refinement.

3. **Stage 3 — Self-distillation** (remaining training time). MoE-mode inference serves as the teacher; dense-like-mode inference serves as the student. The goal is to compress as much of the expert refinement quality as possible back into the micro dense-FFN, maximizing the quality of dense-like mode. Teacher and student share the same micro dense-FFN weights; the teacher path should be treated as a stopped-gradient target to avoid degrading MoE quality.

**Parameter space.** The stage proportions (10–30% / 60–80% / remainder) are rough guidelines. The key principle is that Stage 2 dominates (it trains the most parameters), Stage 1 is long enough for micro dense-FFN convergence, and Stage 3 is a refinement pass. The dense-like batch fraction in Stage 2, consistency loss design, and distillation schedule all require ablation. Optimal proportions depend on model size and dataset.

### 2.5 Block Structure

**Principle (optional).** Organizing the model into macro-blocks of the form `[X MoE layers → 1 dense layer] × Y` (preceded by an input group of dense layers) provides structural benefits, though it may not be strictly necessary for the shared pool mechanism itself.

Benefits of block structure:

1. **Minimal pool block.** The block defines a natural unit for expert pool organization: each block of X MoE layers contains X / sharing_ratio pools. This becomes the minimal allocation unit for exp-GPUs.

2. **Integer divisibility constraint.** X / sharing_ratio must be an integer. For sharing ratio 1:2, valid block sizes are X = 2, 4, 6, 8, ... For ratio 1:3, valid sizes are X = 3, 6, 9, 12, ... This constrains the design space in a useful way, preventing arbitrary configurations.

3. **Dense synchronizer.** The dense layer at the end of each block performs full-width, unrouted processing — effectively summarizing the block's incremental updates before the next block begins. This creates clean phase boundaries and aids pipeline scheduling.

The input group of dense layers (typically 2–4) is already standard practice in MoE architectures and is retained here.

**Note.** Standard MoE architectures without explicit block structure may also support shared pools. The block structure makes the deployment topology cleaner but is not a prerequisite for the core mechanism (micro dense-FFN → shared experts).

---

## 3. Deployment Architecture

The architectural principles above enable a deployment strategy that **separates layer-specific and layer-agnostic model components onto different GPUs**, creating a resource-efficient inference pipeline.

### 3.1 Core Concepts

**Logical replica.** A logical replica consists of the model body — attention weights, micro dense-FFN weights, routers, vocabulary embeddings, and KV-cache — residing on a single GPU. Unlike a conventional replica, it does **not** include the expert pools. Expert pools are external shared resources accessed over the interconnect.

This definition is the key departure from standard deployment: the unit of scaling is the lightweight model body, not the full model.

**Trunk-GPU.** A GPU hosting one logical replica. Contains:
- Model body in FP8 (attention, micro dense-FFN, layer norms)
- Vocabulary embeddings in FP8
- Routers in FP16 (higher precision — routing decisions are critical)
- KV-cache batching in INT8

Because the model body (excluding experts) is a small fraction of total parameters — even for models with hundreds of billions of nominal parameters — the majority of trunk-GPU VRAM is available for KV-cache. This enables large batch sizes or long contexts without additional GPUs.

**KV-cache invariance across regimes.** The KV-cache allocation on a trunk-GPU is designed so that the product of batch size × context length remains constant across operating modes:
- Dense-like: short context × large batch
- MoE effective: long context × moderate batch
- MoE capacity: very long context × small batch

The same trunk-GPU memory layout serves all regimes without reconfiguration.

**Exp-GPU.** A GPU hosting shared expert pools. Design constraints:

1. **Uniform subtype sizing.** If the model requires multiple exp-GPU subtypes (e.g., Type A for early pools, Type B for later pools), each subtype hosts the **same number of pools**. This ensures balanced pipeline throughput — asymmetric subtypes create bottlenecks.

2. **Minimal pool block alignment.** The number of pools on each exp-GPU must be a multiple of the minimal pool block (defined by the block structure). No fractional blocks are allowed.

3. **Virtual slots.** When an exp-GPU hosts more than one minimal pool block, it can serve multiple logical replicas simultaneously by dividing its pools into virtual slots. Each slot is assigned to a different replica in the pipeline. The number of slots is limited both by pool count and by compute capacity — each additional slot shares the same GPU compute.

### 3.2 Conveyor Pipeline

In MoE mode, each logical replica steps through the expert pools sequentially, connecting to different exp-GPU slots at each stage. This creates a **conveyor pipeline** where multiple replicas are in-flight simultaneously, each at a different depth.

**From the replica's perspective:** the forward pass moves through a sequence of slots:

```
Input dense layers → Slot 1 → Slot 2 → ... → Slot N → Output
```

Each slot provides the expert pools for one segment of the model's MoE layers.

**From the exp-GPU's perspective:** at any given moment, each slot is serving a **different** replica. While Replica A is at Slot 3, Replica B is at Slot 2, Replica C is at Slot 1, and so on. The pipeline keeps all slots occupied.

If multiple exp-GPUs of the same subtype exist on the server node, replicas are routed to any available GPU of the needed subtype. The scheduler must maintain depth-indexed queues, support continuous batching, handle variable sequence lengths, and avoid head-of-line blocking. The architectural advantage is that the problem is **structured by pipeline depth** rather than being an arbitrary all-to-all expert-placement problem.

#### Communication Pattern and Load Balance

**Per-layer round-trips.** Within each slot, the trunk-GPU and exp-GPU exchange data on **every MoE layer**, not once per slot. For a slot covering 12 MoE layers (e.g., one macro-block at sharing ratio 1:2 with 6 pools), the per-slot execution looks like:

```
MoE layer 1:  trunk (att → micro-FFN → router) → exp (top-k experts) → trunk
MoE layer 2:  trunk (att → micro-FFN → router) → exp (top-k experts) → trunk
...
MoE layer 12: trunk (att → micro-FFN → router) → exp (top-k experts) → trunk
Dense layer:  trunk only (synchronizer, no exp-GPU access)
```

The dense layer at the end of each macro-block is a **quiet point**: trunk works autonomously, exp-GPU is free, and the next macro-block may address a different exp-GPU subtype.

**Point-to-point, not all-to-all.** In standard expert parallelism (EP), each MoE layer requires an all-to-all scatter/gather across multiple GPUs — every GPU sends tokens to every other GPU holding experts. In this framework, each MoE layer involves a **point-to-point** exchange between exactly two GPUs: one trunk and one exp. Furthermore, all attention weights, all micro dense-FFN weights, all KV-cache reside on a single trunk-GPU — there is no gather across multiple GPUs for any layer-local computation. This is structurally simpler and more predictable than standard EP communication.

**Trunk/exp workload balance.** The trunk-GPU handles significantly more than just its share of MoE layers. Per forward pass, trunk computes: all attention layers (dense + MoE), all micro dense-FFN, all routers, all dense-layer FFN, vocabulary projection, KV-cache management, and layer norms. The exp-GPU computes **only** the top-k expert FFN per MoE layer. In the reference model (per MoE layer): trunk ≈ 6.75d² (attention + micro FFN), exp ≈ 6.0d² (5 experts × 1.2d²). These are roughly comparable per layer — but trunk additionally handles 6 full dense layers (76.5d² total), vocab, and KV-cache, which have no exp-GPU counterpart. This asymmetry is what allows exp-GPU to serve 2–3 slots without becoming a bottleneck: even at 2 slots, the exp-GPU's total work (2 × 6.0d² per MoE layer) remains within range of the trunk's total workload. At 3–4 slots, balance becomes tight — one more reason the practical sweet spot is 2–3 slots.

### 3.3 Server Node Topology: The MoE Node

The natural unit of deployment is the **MoE node**, consisting of:
- One exp-GPU per subtype (or more for throughput)
- Trunk-GPUs equal to the total number of virtual slots across all exp-GPU subtypes

Example: if each exp-GPU has 2 virtual slots and there are 2 subtypes (A and B), the MoE node is:
- 2 exp-GPUs (1 × Type A + 1 × Type B)
- 4 trunk-GPUs (2 slots × 2 subtypes)
- **Total: 6 GPUs serving 4 MoE logical replicas**

On a physical server node (e.g., 8×H100), the remainder after fitting whole MoE nodes is filled with **dense-like replicas** — trunk-GPUs operating in dense-like mode without any exp-GPU dependency. This ensures full GPU utilization with no idle hardware.

### 3.4 GPU-per-Replica Economics

Under a balanced topology, the GPU cost per MoE replica follows:

```
GPU/replica = T + q/N
```

Where:
- `T` — trunk-GPUs required by one logical replica
- `q` — exp-GPUs required in parallel for each subtype/stage
- `N` — virtual slots per exp-GPU

For models where the trunk fits on one GPU (`T=1`) and each subtype is stored on one unsharded exp-GPU (`q=1`), this simplifies to **GPU/replica = 1 + 1/N**:

| Slots per exp-GPU | GPU / MoE replica | Marginal gain |
|---|---|---|
| 1 | 2.0 | — |
| 2 | 1.5 | +0.50 |
| 3 | 1.33 | +0.17 |
| 4 | 1.25 | +0.08 |
| 10 | 1.10 | +0.01/step |

Diminishing returns set in rapidly after 2–3 slots, while compute load per exp-GPU increases linearly. The practical sweet spot is **2–3 virtual slots**, yielding 1.33–1.5 GPU/replica.

The `1 + 1/N` result holds as long as `T=1` and `q=1`. Larger models may require tensor-parallel trunk execution (`T>1`) or sharded expert subtypes (`q>1`), in which case the general `T + q/N` formula applies — the ratio increases but the structural principle remains the same. Increasing the number of sequential depth subtypes (longer pipeline) does not change the ratio; it increases the total node size.

Dense-like replicas always cost exactly **T GPU/replica** (1 GPU in the reference design).

### 3.5 Three Operating Regimes

A single deployed model supports three operating regimes, switchable per-request:

| Regime | Active FFN | Context | Use case |
|---|---|---|---|
| **Dense-like** | Micro dense-FFN only (top-k=0) | Short, large batch | High-throughput simple tasks |
| **MoE Effective** | Micro dense-FFN + top-k experts (active = dense) | Medium, moderate batch | Standard inference |
| **MoE Capacity** | Micro dense-FFN + ~1.5× top-k experts | Long, small batch | Complex reasoning, long context |

In the capacity regime, top-k is intentionally increased beyond the constraint value (e.g., from 5 to 8), giving MoE layers more active compute than dense layers. This trades throughput for quality, and typically restricts each exp-GPU to one active slot.

The KV-cache invariance property ensures that all three regimes use the same trunk-GPU memory layout (batch × context = constant), enabling seamless switching.

---

## 4. Benefits

### 4.1 VRAM Efficiency (Direct Arithmetic)

Expert pools constitute 80–90% of a conventional MoE model's parameters. Shared pools compress this by the sharing ratio (e.g., 2× for 1:2 sharing). Trunk/exp disaggregation further ensures that these compressed pools are stored **once** per exp-GPU and shared across all replicas in the pipeline — not duplicated per replica.

The result: the per-replica VRAM cost is dominated by the model body (10–20% of nominal parameters) plus KV-cache, not by experts. This is what enables 1.33–1.5 GPU/replica even for models with hundreds of billions of nominal parameters.

### 4.2 Unlocked Design Space

In conventional MoE, three key axes — model width (d_model), model depth (number of layers), and expert capacity (expert size) — compete for the same parameter budget. Increasing any one axis requires shrinking the others.

Shared pools **break this trade-off** by compressing the dominant cost (expert storage). The freed budget can be invested along any axis:
- **Wider models** (larger d_model) for richer representations
- **Deeper models** (more layers) for more processing stages
- **Larger experts** for more expressive refinement
- Or all three simultaneously — achieving configurations that would require multi-trillion-parameter budgets in conventional MoE, at a fraction of the nominal size

Concretely: the reference model (Section 6) has d_model = 5,120, 54 attention layers, and 160 experts per pool — hyperparameters typical of conventional MoE models in the 400–600B range. Yet its nominal size is 132B, because shared pools compress the expert storage that normally dominates the parameter count. The model effectively has the "body" of a much larger model class. This is the connection to C-factor (Section 5): the metric attempts to quantify *which* larger class the model's quality approaches, based on how much of the shared pool each layer actually utilizes.

This is arguably the most important benefit: shared pools do not just make the same model cheaper to deploy — they make **previously impossible model configurations accessible**.

### 4.3 Structured Pipeline Integration

The trunk/exp disaggregation and conveyor pipeline arise naturally from the shared pool architecture. Because expert pools are not bound to specific layers, placing them on separate GPUs and sharing them across replicas is architecturally clean — the pipeline stages are known at design time and indexed by depth.

Conventional per-layer MoE models can also be disaggregated (and prior systems have demonstrated attention/expert separation). The advantage here is co-design: shared pools, macro-block boundaries, slot geometry, top-k, and trunk/exp compute are selected together rather than retrofitted onto independently designed per-layer expert sets.

---

## 5. C-factor: Pool Breadth and Sharing Efficiency

The C-factor estimates the effective quality of a shared-pool model by combining two measurable quantities: how many experts each layer actually uses, and how well shared experts perform compared to dedicated per-layer ones.

### 5.1 Observable Pool Breadth

For each layer `l`, two complementary measures:

**Thresholded coverage** — the fraction of experts receiving meaningful traffic above a threshold `τ` over a measurement window `W`:

```
C_l(τ, W) = |{e : p_{l,e} ≥ τ}| / E
```

**Entropy-effective expert count** — discounts experts that are technically used but receive negligible routing mass:

```
N_eff,l = exp(H(p_l))     where H is routing entropy
```

Example: if E=256 and thresholded coverage is 90%, the layer has meaningful access to ~230 experts. This establishes **accessible breadth** — how many experts the layer can draw from.

### 5.2 Sharing-Transfer Efficiency

Pool breadth alone does not determine quality. A shared expert serving multiple layers may not be as effective as a dedicated per-layer expert. The **sharing-transfer efficiency** `η_share ∈ [0, 1]` captures this gap:

```
E_equiv,l = η_share × N_eff,l
```

`η_share = 1` would mean shared experts are fully equivalent to per-layer ones. The framework predicts that the micro dense-FFN architecture pushes `η_share` close to 1 (because experts handle only generic refinement), but this must be measured, not assumed.

A complete C-factor report should include: thresholded coverage, entropy-effective count, pairwise pool overlap between sharing layers, measured `η_share` against per-layer baselines, and confidence intervals across layers and seeds.

### 5.3 Testable Prediction

The framework makes a specific falsifiable prediction: increasing micro dense-FFN capacity and using force-then-relax routing will increase `η_share` by moving depth-specific work out of the experts. To test this:

1. Train matched shared-pool and per-layer baselines at equal active compute.
2. Measure routing breadth and overlap.
3. Estimate `η_share` from the quality gap.
4. Ablate micro dense-FFN width, sharing ratio, and router curriculum.
5. Test whether C-factor predicts quality across scales.

Until such calibration exists, the C-factor is a **diagnostic and a prediction**, not a validated quality estimator.

---

## 6. Reference Model: 132B on H100

The following model instantiates the principles above at a specific point in the parameter space. It is presented as a **concrete example**, not as the recommended configuration. Different deployment targets, hardware generations, and quality requirements would produce different designs using the same principles.

### 6.1 Architecture

**Layer structure:** 2 dense → [12 MoE → 1 dense] × 4

| Component | Count |
|---|---|
| Dense layers | 6 (2 input + 4 block-terminal) |
| MoE layers | 48 |
| Total attention layers | 54 |
| Shared expert pools | 24 (sharing ratio 1:2) |

### 6.2 Dimensions

| Parameter | Value |
|---|---|
| d_model | 5,120 |
| Attention heads (Q / KV) | 40 / 5 (GQA 8:1) |
| Head dimension | 128 |
| Vocabulary | 131,072 tokens |

### 6.3 FFN Design

**Dense-layer FFN** (SwiGLU): d → 3.5d → d

**MoE-layer components:**
- Micro dense-FFN (SwiGLU): d → 1.5d → d — layer-specific, always active
- MLP Router: d → 512 → 160 (~2.7M params) — layer-specific

**Shared pool** (per pool of 2 MoE layers):
- 160 experts, each SwiGLU: d → 0.4d → d
- Top-k: 5 (in effective regime)

**Constraint verification:** 1.5d + 5 × 0.4d = 3.5d = inter_dense ✓

**Active layer size:**
- Dense layer: attention (2.25d²) + FFN (10.5d²) = 12.75d²
- MoE layer (effective): attention (2.25d²) + micro FFN (4.5d²) + 5 experts (5 × 1.2d² = 6.0d²) = 12.75d² ✓

### 6.4 Model Size

| Metric | Value |
|---|---|
| Nominal parameters | ~132B |
| Dense-like active (per token) | ~11.2B |
| MoE effective active (per token) | ~18.85B |

### 6.5 Context and KV-Cache

**KV-cache per token** (INT8): 69,120 bytes

| Regime | Context | Batch | KV-cache budget | Product |
|---|---|---|---|---|
| Dense-like | 8,192 | ×64 | ~45.3 GB | 524,288 |
| MoE effective | 65,536 | ×8 | ~45.3 GB | 524,288 |
| MoE capacity | 131,072 | ×4 | ~45.3 GB | 524,288 |

Cold retrieval: 262K tokens in 512–2,048 chunks (type-dependent).

Output limits: dense ≤2,048 tokens; MoE ≤8,192 tokens + optional 24,576 think tokens.

### 6.6 H100 Deployment

**Trunk-GPU (H100, 80 GB):**

| Component | VRAM |
|---|---|
| Model body (FP8) | 10.5 GB |
| Routers (FP16) | 0.26 GB |
| Vocabulary (FP8) | 0.67 GB |
| KV-cache budget | 45.28 GB |
| **Reserve** | **23.29 GB** |

**Exp-GPU (H100, 80 GB):**

| Component | Value |
|---|---|
| Pool size | ~5.03 GB |
| Pools per exp-GPU | 12 (2 minimal blocks of 6) |
| VRAM used | ~60.4 GB |
| Virtual slots | 2 (6 pools per slot) |
| Subtypes | A (pools 1–12) and B (pools 13–24) |

**MoE node:** 2 exp-GPU (Type A + Type B) + 4 trunk-GPU = **6 GPU, 4 MoE replicas** (1.5 GPU/replica)

**8×H100 block layout:**
- 1 MoE node: 4 MoE replicas (6 GPUs)
- 2 dense-like replicas (2 GPUs)
- **Total: 6 logical replicas on 8 GPUs**
- MoE concurrent sequences (batch 8): 32 in-flight, 8 per pipeline wave
- Dense-like concurrent sequences (batch 64): 128

### 6.7 Why These Specific Values

Every number above is one point in the parameter space defined by the principles:

- **inter_micro = 1.5d** rather than 1.25d — at d_model = 5,120, a 1.25d intermediate (6,400 dimensions) may be too narrow for robust layer-specific processing. 1.5d (7,680 dimensions) provides comfortable headroom. Empirically, this may prove conservative or insufficient.
- **inter_exp = 0.4d** rather than 0.375d — at this d_model, 0.375d would produce 1,920-dimensional experts; 0.4d (2,048) is a cleaner power-of-2 size and fits the constraint with integer top-k. Additionally, (3.5d - 1.5d) / 0.375d = 5.33, which is not an integer; 0.4d yields exactly 5.
- **Sharing ratio 1:2** — conservative. Higher ratios (1:3, 1:4) would compress expert storage further but increase the risk of inter-layer interference. 1:2 is a safe starting point.
- **E = 160** — the router's output dimension. Balances pool diversity against routing complexity.
- **inter_dense = 3.5d** — follows from the constraint. Wider than typical but necessary to maintain uniform active compute with a meaningful micro dense-FFN.

A model targeting H200 (141 GB) with the same principles could fit more pools per exp-GPU (wider blocks, fewer subtypes), increase d_model, or allocate more KV-cache — the principles remain the same, only the point in parameter space shifts.

---

## 7. Open Questions

1. **Expert portability.** To what extent does the sequential micro dense-FFN reduce depth dependence in expert functions? This should be tested through representation similarity analysis, cross-layer expert substitution experiments, and gradient-conflict measurements.

2. **Uniform FFN budget vs. unconstrained.** Does the constraint `inter_micro + k × inter_exp = inter_dense` improve quality or systems balance relative to allowing MoE layers to have different active widths? Both validation quality and measured stage time should be reported.

3. **Micro dense-FFN sizing.** What is the minimum inter_micro that provides sufficient layer-specific processing to make expert sharing safe? The theoretical range is 1.25–2.0×d, but the practical sweet spot may be narrower — or may vary with d_model.

4. **Training stage proportions.** Optimal proportions for the three training stages likely depend on model size, dataset, and the desired quality balance between dense-like and MoE modes. The dense-like batch fraction in Stage 2 and self-distillation schedule require ablation.

5. **Force-then-relax routing.** Which auxiliary loss and decay schedule produce broad initial exploration without permanently suppressing useful specialization?

6. **C-factor calibration.** Can `η_share` and `N_eff` predict the quality gap to matched per-layer baselines across model sizes, sharing ratios, and tasks?

7. **Router architecture and overhead.** MLP routers (d → 512 → E) provide richer routing decisions than linear routers but add latency. The wall-clock cost needs profiling at scale.

8. **Pipeline scheduling.** How large are pipeline bubbles under realistic mixtures of prefill, decode, variable output lengths, and tool calls? Depth-indexed continuous batching must be evaluated under production conditions.

9. **Trunk/exp load balance.** The effective and capacity regimes should be profiled separately — attention and KV-cache may make trunk execution bandwidth-bound, while exp-GPUs depend on grouped-GEMM occupancy and routing skew.

10. **Scaling behavior.** The framework predicts that shared pools become safer at larger scales due to deeper models, wider hidden states, and more aggressive GQA ratios. This requires controlled scaling experiments, not parameter-count extrapolation.

11. **Cold retrieval architecture.** The extended-context mechanism is orthogonal to shared pools and should be evaluated separately.

---

## 8. Conclusion

Shared expert pools, as presented in this framework, rest on a single architectural insight: **if layer-specific processing is handled by a dedicated, always-active component (micro dense-FFN), then expert pools do not need to be layer-specific — and can be shared across layers and across replicas.**

This insight produces a coherent system:
- A **layer design** where experts refine rather than replace dense processing
- A **training recipe** where dense-like and MoE capabilities are developed in staged coordination
- A **deployment architecture** where expert sharing enables 1.33–1.5 GPU per replica (under `T + q/N` accounting, with `T=1`, `q=1` for the reference design)
- A **diagnostic framework** (C-factor) that separates observable pool breadth from empirically calibrated sharing efficiency
- **Three operating regimes** (dense-like, effective, capacity) on the same hardware with the same weights

The framework is not tied to a specific model size, hardware generation, or hyperparameter configuration. The reference model (132B on H100) demonstrates one instantiation; the principles apply across the design space.

All claims are theoretical. The framework produces specific, testable predictions — about quality retention, about C-factor accuracy, about the viability of the training recipe — that can and should be validated experimentally. The structural arguments are strong enough to warrant that investment.

---

## References

1. UniPool: A Globally Shared Expert Pool for Mixture-of-Experts. arXiv:2605.06665
2. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. arXiv:2401.06066
3. Tying the Loop: Tied Expert Layers in Mixture-of-Experts Language Models. arXiv:2606.16825
4. MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism. arXiv:2504.02263
5. Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism. arXiv:2512.21487
6. ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs. arXiv:2607.18002
7. REAM: Merging Improves Pruning of Experts in LLMs. arXiv:2604.04356

---

*No experimental validation has been performed. All architectural principles are presented as testable hypotheses. All numerical parameters are starting points for empirical search. The reference model is one instantiation of the framework, not a final design.*
