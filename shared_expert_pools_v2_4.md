# Shared Expert Pools: An Architectural Framework for Efficient Mixture-of-Experts Design and Deployment

**Aleksei Ketslakh — Architectural Framework Proposal, version 2.4 (September 2026)**

---

## Abstract

This document presents **shared expert pools** — a set of architectural principles for Mixture-of-Experts (MoE) models in which groups of adjacent layers draw from the same pool of experts instead of maintaining independent per-layer expert sets.

Unlike prior approaches to expert sharing (e.g., UniPool) that introduce layer-conditioning mechanisms to help shared experts adapt to different depths, this framework takes the opposite approach: it ensures that **experts do not need layer-specific adaptation at all**. A sufficiently large layer-specific micro dense-FFN absorbs all depth-dependent processing before experts are invoked, leaving only generic refinement work for the shared pool — work that is inherently portable across layers.

The framework defines:
- **Structural principles** for MoE layer design (sequential micro dense-FFN → routed experts, uniform active compute constraint), formalized through a four-parameter design space `{D, λ, k, r}` with a testable portability hypothesis linking micro dense-FFN capacity to safe sharing depth
- **A two-stage training architecture** (D/E pretraining; D/E/C post-training; continuous micro dense-FFN optimization; force-then-relax routing; post-training variable-k collaboration; hierarchical C→E→D self-distillation)
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

**Nested operating modes.** The sequential design supports one expert-disabled path and two expert-enabled paths:
- **Dense-like (D)** — processing ends after the micro dense-FFN (`top-k = 0`). No expert lookup and no communication with exp-GPUs.
- **MoE Effective (E)** — the router activates `k_E` experts while preserving the uniform active-compute constraint.
- **MoE Capacity (C)** — the router activates `k_C > k_E` experts from the same pool, spending additional compute on deeper refinement.

All three modes use the same model weights. They form nested execution envelopes around the same continuously optimized layer-specific core; they differ in the amount of routed refinement made available.

### 2.2 Micro Dense-FFN: The Layer-Specific Workhorse

**Principle.** The micro dense-FFN is not a "shared expert" in the DeepSeek sense — a small always-on expert running alongside routed ones. It is a **dense-type FFN with expanding intermediate dimension** (inter > 1×d_model), providing full-width nonlinear processing of the residual stream.

The name "micro" refers to it being narrower than the full dense-layer FFN — not to it being a minor component. In this framework it is a primary layer component.

Key properties:

1. **Expanding intermediate dimension.** Unlike expert FFNs (which have contracting intermediates, typically 0.3–0.5×d), the micro dense-FFN has an intermediate dimension larger than d_model. This makes it a genuine dense-layer-class processor, not a miniature expert. The expansion ratio is smaller than in full dense layers, but the processing is qualitatively the same: project up, apply nonlinearity, project down.

2. **Layer-specific ownership.** Each MoE layer has its own micro dense-FFN with independent weights. This is what provides layer identity to the architecture — the micro dense-FFN is the component that "knows" it is at a particular depth. Shared experts do not need this knowledge because their job begins after it has already been applied.

3. **Significant compute share.** The micro dense-FFN accounts for a substantial fraction of the MoE layer's active FFN compute — not a minor addition, but a primary component. If the micro dense-FFN were small (as in DeepSeek's shared expert approach), it could not absorb enough layer-specific work, and experts would be forced to compensate — making sharing problematic.

4. **Active and trainable throughout the training lifecycle.** Unlike routed experts (which see only the tokens routed to them), the micro dense-FFN processes every token at every MoE layer. It is not a base trained once and later frozen: it remains trainable in every regime that is active—D/E during pretraining and D/E/C during post-training. D teaches it to remain self-sufficient; E and C teach it to produce a complete representation that is also an effective substrate for progressively richer expert refinement. Stopped-gradient teacher branches used for distillation are the sole procedural exception: they supply targets rather than parameter updates.

**Parameter space.** The micro dense-FFN's intermediate dimension is expected to fall in the range of 1.25–2.0×d, corresponding to roughly 30–50% of the MoE layer's total active FFN compute. The lower bound (1.25×d) may prove sufficient for layer-specific processing; the upper bound (2.0×d, which in SwiGLU is roughly equivalent to 3.0×d in GeLU — a standard value) provides generous capacity. The actual sweet spot is an empirical question.

### 2.3 The Design Space: {D, λ, k, r}

#### 2.3.1 The Uniform Compute Constraint

**Principle.** In the effective MoE regime, the total active FFN width of every layer in the model — whether dense or MoE — should be uniform:

```
inter_micro + k × inter_exp = inter_dense
```

This constraint means that **every layer performs the same nominal amount of active FFN computation**, regardless of whether it is a dense layer or an MoE layer. The MoE layer simply splits this budget between two components: layer-specific processing (micro dense-FFN) and generic refinement (routed experts).

**Important caveat.** The constraint equalizes nominal FFN arithmetic and parameter access, not wall-clock latency. A routed MoE layer still includes router execution, dispatch, inter-GPU communication, grouped GEMMs, and result merging. Equal arithmetic width does not guarantee equal runtime. The constraint is a first-order balancing target that simplifies hardware co-design and narrows the search space; actual stage-time parity must be profiled.

This is a **design constraint, not an observation**. It is imposed deliberately to achieve:
- Comparable nominal FFN work across dense and MoE layers
- More predictable pipeline balancing
- Clean separation between "how much compute" (fixed) and "what kind of compute" (dense vs. split)

#### 2.3.2 Parametric Formulation

The constraint above, together with the sharing ratio, defines a design space with **four fundamental parameters**:

| Symbol | Name | Definition |
|---|---|---|
| `D` | Total FFN budget | `inter_dense` (the dense-layer FFN width) |
| `λ` | Micro share | `M / D`, where `M = inter_micro` |
| `k` | Effective top-k (`k_E`) | Number of experts activated per token in the effective regime |
| `r` | Sharing ratio | Number of MoE layers sharing one expert pool |

All derived quantities follow algebraically:

```
M = λD                    # micro dense-FFN width
R = (1 − λ)D              # routed budget (total active expert width)
e = (1 − λ)D / k          # single expert width — a consequence, not a free parameter
```

This reduces the hyperparameter space from five apparent degrees of freedom (`inter_dense`, `inter_micro`, `inter_exp`, `k`, `r`) to four (`D`, `λ`, `k`, `r`), with `inter_exp` fully determined by the others. The constraint `D = M + ke` is automatically satisfied by construction.

Under the `{D, λ, k, r}` parameterization, effective-regime `k` is a **design variable**, while expert width `e` is derived from it. The derived width must map to a hardware-efficient integer dimension. Capacity-mode `k_C` is a separate post-training and deployment choice and may intentionally exceed `k_E`.

**Example configurations** (all satisfying the constraint):

| D | λ | M | k | e | Notes |
|---|---|---|---|---|---|
| 3.5d | 0.43 | 1.5d | 5 | 0.4d | Reference model (Section 6) |
| 3.75d | 0.47 | 1.75d | 5 | 0.4d | Stronger micro, same expert size |
| 3.75d | 0.47 | 1.75d | 8 | 0.25d | More, smaller primitives |
| 4.25d | 0.47 | 2.0d | 6 | 0.375d | Wide budget, large micro |
| 4.25d | 0.47 | 2.0d | 8 | 0.281d | Wide budget, fine-grained experts |

Several observations:

1. **`λ ≈ 0.47` is a candidate design ratio, not an established invariant.** Several illustrative configurations recur near a split of ~47% layer-specific and ~53% routed compute. Because these points were constructed rather than independently measured, the recurrence is only a hypothesis-generating observation. Whether it corresponds to an empirical optimum must be tested.
2. **`e` is not a design choice — it is a consequence.** Choosing `D`, `λ`, and `k` fully determines the expert width. This is a key simplification: the designer selects the budget, the micro share, and the granularity of composition — the expert size follows.
3. **`k` and `e` form a granularity axis** at constant `R = (1−λ)D`. Increasing `k` while holding `R` constant shrinks each expert and increases their number in the active set. The routed budget is invariant: `ke = R = const`. This means `k` controls the **granularity of composition** — fewer large experts vs. more small primitives — without changing the total routed compute.
4. **Dense layers are wider than typical.** The constraint produces D values of 3.25–4.25×d (vs. 2.67–3.5×d in standard architectures). This is a necessary consequence of maintaining uniform active compute while giving the micro dense-FFN a meaningful expansion ratio.
5. **Top-k is expected to remain moderate** (roughly 4–8 in the effective regime). At fixed `D` and `λ`, increasing `k` does not force `λ` downward; it narrows each expert according to `e = (1−λ)D/k`. The practical upper bound is therefore reached when individual experts become too narrow to learn useful, hardware-efficient refinement primitives—not when the micro dense-FFN necessarily ceases to expand.

#### 2.3.3 Expert Semantics: Composable Primitives

The parametric formulation reveals an important shift in the role of experts. Conventional MoE experts are generally expected to perform a substantially more autonomous transformation. In this framework, that responsibility is intentionally divided between the layer-specific micro dense-FFN and multiple routed corrections; the distinction is functional rather than dependent on any single conventional expert-width range.

In this framework, experts are **composable refinement primitives**: "provide one specialized nonlinear correction that the layer-specific processor decided to use." The expert width is deliberately small (0.25–0.4×d), making each expert a narrow basis function rather than an independent processor.

The layer's output can be expressed as a basis-like decomposition:

```
F(x, l) ≈ Micro_l(x) + Σ_{i∈TopK} g_i(x, l) · E_i(Micro_l(x))
```

where `Micro_l` provides the primary transformation and each `E_i` provides a small learned correction. The smaller each `E_i`, the more this resembles a **composition of primitives** rather than a selection among alternative processors. This semantic shift is what makes sharing natural: a "correction tool" can serve multiple workstations; a "self-contained processor" cannot.

This is consistent with the broader trend toward fine-grained expert decomposition (DeepSeekMoE), where activating more smaller experts has been shown to outperform fewer larger ones at equal compute. The shared pool framework provides an architectural motivation for this observation: smaller experts are more portable precisely because they do less layer-specific work.

#### 2.3.4 The Portability Hypothesis

**Hypothesis.** The depth-specific residual in expert computation decreases monotonically with micro dense-FFN capacity.

Formally, represent the work each expert performs at layer `l` as:

```
F_l = G + D_l
```

where `G` is a generic (depth-invariant) component and `D_l` is a depth-specific component. After micro dense-FFN processing with capacity `M`:

```
F_l^expert = G + D_l^residual(M)
```

The hypothesis asserts:

```
∂D_l^residual / ∂M < 0
```

That is: increasing the micro dense-FFN's capacity absorbs more of the depth-specific work, leaving experts with a progressively more generic (and therefore more portable) task.

In the limit: as `M → D` (the micro dense-FFN absorbs the entire FFN budget), `D_l^residual → ε_l ≈ 0`. Residual states may remain strongly depth-dependent, but the **transformation demanded from the expert branch** becomes approximately depth-invariant. Expert portability should therefore approach its maximum even though expert inputs are not identical across layers.

**This is testable.** Train controlled models across a factorial grid of `D/d` and `λ`, comparing runs at matched active compute within each fixed-`D/d` slice. This varies `M/d` without treating fractional share and normalized absolute width as interchangeable. Measure inter-layer interference via cross-layer expert substitution accuracy (swap experts between layers sharing a pool and measure quality degradation). If the portability hypothesis holds, degradation should decrease monotonically with `M/d` when relevant secondary variables are controlled.

#### 2.3.5 The Sharing-Portability Bound (Conjecture)

If the portability hypothesis holds, then micro dense-FFN capacity is a primary determinant of the **maximum safe sharing ratio** — the deepest sharing that preserves quality above a target threshold. The normalized absolute width `m = M/d`, rather than `λ` alone, must be retained because identical micro shares can correspond to different absolute expansion ratios when `D/d` changes.

```
r_max = f(M/d, λ, k, E, η_target; training, scale, hardware, ...)
```

where `η_target` is the minimum acceptable sharing-transfer efficiency (Section 5.2), `E` is pool breadth, and training, scale, and hardware denote potential contextual dependencies. Derived values such as `D/d` and `e/d` need not be listed separately because they follow from `M/d`, `λ`, and `k`. Holding the other variables fixed, `f` is expected to be:
- **Monotonically increasing** in normalized micro capacity `M/d` (more layer-local capacity → safer sharing)
- **Saturating** (diminishing returns at high micro capacity—once the micro dense-FFN handles nearly all depth-specific work, further increases add little portability)

Conjectured values at `η_target = 0.95`:

| D/d | λ | M/d | Conjectured r_max |
|---|---|---|---|
| 3.75 | ~0.40 | 1.50 | 2 |
| 3.75 | ~0.47 | 1.75 | 3 |
| 4.25 | ~0.47 | 2.00 | 4 |

These specific numbers are speculative. In particular, the final two rows intentionally show that equal `λ` need not imply equal `r_max`: increasing `D/d` also increases `M/d`. The **conditional form** of the relationship (monotonic and saturating in `M/d` when other variables are fixed) follows from the portability hypothesis; both the values and the significance of the secondary dependencies require empirical calibration.

#### 2.3.6 Micro Dense-FFN as Storage-Compute Control Variable

The portability hypothesis and sharing-portability bound together reveal that `inter_micro`—described jointly by `M/d` and `λ`—is not merely a hyperparameter. It is the **primary control variable for the architecture's storage-compute tradeoff**.

Increasing `λ` at fixed total FFN budget `D` simultaneously:
- Reallocates a larger share of fixed nominal FFN compute from routed experts to the layer-local trunk path
- Decreases `(1−λ)`, reducing per-pool storage (direct effect)
- May increase `r_max`, allowing more aggressive sharing (indirect effect via portability, subject to the dependencies above)

The storage effects are **multiplicative**. Total expert storage scales as:

```
S_total ∝ E · (1−λ) · D · N_MoE / (k · r)
```

A modest increase in `λ` reduces `(1−λ)` (numerator) *and* may permit a larger `r` (denominator). The combined effect can be substantial: increasing `M/d` by roughly 17% (1.5 → 1.75) may enable `r: 2→3`, yielding a 33% reduction in pool count *on top of* the direct per-pool shrinkage. This numerical example is conjectural, not a measured exchange rate.

This is potentially a favorable exchange: **a modest reallocation toward active layer-local parameters may remove a large number of inactive nominal expert parameters**, without changing the total nominal active FFN budget. The constraint `D = M + ke` absorbs the shift by reducing the routed share through `e`, `k`, or both. The actual wall-clock effect is hardware-dependent because trunk and expert work have different communication and scheduling costs.

**Parameter space.** A plausible initial search region is `D ∈ [3.25d, 4.25d]`, `λ ∈ [0.35, 0.55]`, `k_E ∈ [4, 8]`, `r ∈ [2, 4]`. The micro dense-FFN's share of nominal active FFN compute is therefore 35–55% in this proposed region. Different model sizes and deployment targets will favor different points, but the parametric relationships `{D, λ, k, r}` and the constraint `D = λD + ke` hold throughout.

### 2.4 Training Recipe

**Principle.** Training separates acquisition of the model's internal world representation from learning how to spend progressively larger inference budgets on that representation:

In this section, **D-mode**, **E-mode**, and **C-mode** denote execution regimes. The scalar `D` in the architectural design space continues to denote total FFN width; the meanings are disambiguated by context.

```
Pretraining:   D + E
Post-training: D → E → C
Distillation:  C → E → D
```

The three regimes are:

```
D: y_D = f(x; k = 0)              # dense-like; complete low-cost computation
E: y_E = f(x; k = k_E)            # effective expert refinement
C: y_C = f(x; k = k_C > k_E)      # capacity refinement with additional experts
```

Pretraining uses D and E because both contribute directly to representation learning at controlled cost. C uses no additional stored weights relative to E; it only activates more experts from the same pool. Paying this additional compute throughout pretraining is therefore not assumed to increase knowledge capacity proportionally. Instead, C is introduced during post-training, where the model explicitly learns to make productive use of the larger active set.

#### Continuous Micro Dense-FFN Training Invariant

The micro dense-FFN is the shared computational trunk of all three regimes. It is active on every token at every MoE layer and remains trainable in every non-teacher optimization path throughout both pretraining and post-training:

```
D: Micro_l learns a self-sufficient approximation
E: Micro_l co-adapts with TopK_{k_E} expert refinement
C: Micro_l co-adapts with TopK_{k_C} expert refinement
```

D batches do not uniquely "train the micro" while E and C merely train experts. Rather, D imposes a lower bound on micro autonomy, while E and C require the same micro to produce representations that are both complete and efficiently refinable. Conceptually, the target decomposition is:

```
Micro_l = complete layer-local approximation
Experts = routed error correction and refinement
```

This invariant prevents the micro dense-FFN from degenerating into a preprocessing layer that assumes experts will complete its work. It also gives downward distillation a shared accumulation path: capabilities learned under richer expert execution can migrate into micro dense-FFN, attention, and residual representations and thereby become available to cheaper modes.

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

The target utilization range (80–95%) is a structural expectation, not a universal optimum. Utilization below 80% may indicate that the shared pool is oversized, the routing curriculum is too weak, or depth-specific work remains in the expert branch. Utilization above 95% with little layer differentiation may indicate that the pool is too small, the router is underpowered, or the auxiliary objective remains too strong.

#### Pretraining: D/E Knowledge Acquisition

Pretraining contains an initial D-only warm-up followed by mixed D/E optimization:

1. **D-only warm-up.** Attention and micro dense-FFN are active; routers and experts are inactive. The model learns to produce a rough but complete computation without routed assistance. This establishes a usable lower-cost path before expert co-adaptation begins.

2. **Mixed D/E pretraining.** Routers and experts are activated at `k_E`, while a persistent fraction of batches continues in D. The micro dense-FFN remains trainable in both paths. E batches teach expert specialization and collaboration; D batches prevent the shared trunk from outsourcing essential computation to the routed branch.

The principal pretraining objective can be written schematically as:

```
L_PT = w_D L_D + w_E L_E + L_router + L_balance
```

where `w_D` remains nonzero after E is introduced. The appropriate D/E batch mixture and warm-up duration are empirical questions. The previous rough range of 10–30% remains a plausible starting point for D-only warm-up, not a fixed prescription.

Capacity mode is deliberately excluded from routine pretraining. Limited C probes may be used for diagnostics or ablations, but the default hypothesis is that pretraining should spend compute on broad D/E knowledge acquisition rather than on repeatedly activating a larger fraction of already-present expert parameters.

#### Post-training: D → E → C Computational Curriculum

Post-training teaches the shared weights to solve the same task at three levels of computational thoroughness:

1. **D — construction.** The model must produce a complete answer or solution with the expert branch disabled. Quality may be lower than in E or C, but the output must not be structurally incomplete or depend on unavailable experts.
2. **E — refinement.** The effective expert set learns to correct, verify, specialize, and polish an already functional computation. Experts should improve the result rather than reconstruct it from an inadequate trunk representation.
3. **C — deep refinement.** A larger active subset from the same pool learns additional cross-expert combinations, verification, and fine-grained correction. C is not assumed to know a categorically different world model; it is trained to extract, compare, and apply shared knowledge more thoroughly.

The arrow denotes curriculum order, not permanent retirement of earlier modes. Once introduced, lower modes continue to receive task and consistency batches so that post-training does not improve C at the expense of D or E. This yields a nested quality hierarchy:

```
D = M
E = M + Δ_E
C = M + Δ_E + Δ_C
```

where `M` denotes the shared micro-based path and the deltas denote progressively broader routed refinement. The equations are semantic decompositions; in implementation E and C use different top-k subsets from the same ranked pool rather than separately parameterized branches.

#### Variable-k Collaboration and Nested Ranking

Simply training at `k_E` and activating `k_C > k_E` at inference is not sufficient. Experts ranked below the effective cutoff may be individually useful yet untrained to collaborate with the higher-ranked set. Capacity training must therefore expose the model to the intended range of active expert counts during post-training.

For router logits with a stable ordering, the selected sets are naturally nested:

```
TopK_{k_1} ⊂ TopK_{k_2} ⊂ ... ⊂ TopK_{k_C},  where k_1 < k_2 < ... < k_C
```

The training requirement is stronger than set inclusion: each increment must provide positive marginal utility. Variable-k post-training should sample D (`k=0`), E (`k=k_E`), C (`k=k_C`), and optionally intermediate values. Ranking, consistency, and marginal-gain objectives may be used to discourage additional experts from producing redundant or harmful corrections. A practical initial range is `k_C ≈ 1.5–2k_E`, subject to expert width, grouped-GEMM efficiency, and diminishing returns.

#### Hierarchical Self-Distillation: C → E → D

The richest execution of the same model supplies targets for cheaper execution:

```
C teaches E
E teaches D
```

This staged transfer is preferable to relying only on direct C→D distillation because the quality and compute gap between adjacent modes is smaller. A schematic objective is:

```
L_PTune = L_C + L_E + L_D
        + β_CE L_distill(y_E, stopgrad(y_C))
        + β_ED L_distill(y_D, stopgrad(y_E))
        + β_CD L_distill(y_D, stopgrad(y_C))
```

with `β_CD` optional and normally smaller than the adjacent-mode coefficients. Distillation may operate on output distributions, selected intermediate representations, or both. The teacher forward pass is stopped-gradient, but the micro dense-FFN remains trainable through the C, E, and D task paths and through the student side of the consistency losses.

Because all modes share weights, this is not a separate large teacher compressing into independent student checkpoints. It is **richer execution teaching cheaper execution within one parameter set**. Care is required to prevent destructive gradient interference: regime-balanced batches, gradient surgery, delayed consistency losses, or exponential-moving-average teacher targets are candidate implementation choices rather than assumed necessities.

**Training parameter space.** Warm-up duration, D/E pretraining mixture, D/E/C post-training mixture, `k_C/k_E`, curriculum overlap, distillation coefficients, teacher update rule, and router-loss decay all require ablation. The architectural commitments are narrower: C is trained as a post-training compute mode; D remains a complete path; E and C refine the same continuously optimized micro-based core; and richer execution transfers quality downward without becoming a separate model.

### 2.5 Block Structure

**Principle (optional).** Organizing the model into macro-blocks of the form `[X MoE layers → 1 dense layer] × Y` (preceded by an input group of dense layers) provides structural benefits, though it may not be strictly necessary for the shared pool mechanism itself.

Benefits of block structure:

1. **Minimal pool block.** The block defines a natural unit for expert pool organization: each block of X MoE layers contains X / sharing_ratio pools. This becomes the minimal allocation unit for exp-GPUs.

2. **Integer divisibility constraint.** X / sharing_ratio must be an integer. For sharing ratio 1:2, valid block sizes are X = 2, 4, 6, 8, ... For ratio 1:3, valid sizes are X = 3, 6, 9, 12, ... This constrains the design space in a useful way, preventing arbitrary configurations.

3. **Dense synchronizer.** The dense layer at the end of each block performs full-width, unrouted processing — effectively summarizing the block's incremental updates before the next block begins. This creates clean phase boundaries and aids pipeline scheduling.

The input group of dense layers (typically 2–4) is already standard practice in MoE architectures and is retained here.

**Compatibility with hybrid attention.** The block structure `[X MoE → 1 dense]` creates a natural correspondence with hybrid attention architectures (e.g., MiMo): dense layers serve as global-attention synchronization points (full attention), while MoE layers use local/sliding-window attention. This mapping requires no special adaptation — it follows from existing roles: the dense layer already performs full-width unrouted processing of the entire residual stream, making it the natural candidate for global context aggregation. At block-level ratios of 1:6–1:12, a local attention window of 2,048–4,096 tokens provides sufficient mid-range context between global synchronization points. This is an orthogonal optimization that composes with shared pools without architectural changes.

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

Capacity mode is a trained operating point, not an inference-time extrapolation. Its additional experts have been exposed to variable-k collaboration during post-training, so increasing top-k invokes a learned refinement regime rather than merely adding lower-ranked experts to a computation optimized only at `k_E`.

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

Every number above is one point in the parametric design space `{D, λ, k, r}` defined in Section 2.3.2. The reference model sits at `D = 3.5d`, `λ = 0.43`, `k = 5`, `r = 2`:

- **λ = 0.43 (inter_micro = 1.5d)** rather than 1.25d — at d_model = 5,120, a 1.25d intermediate (6,400 dimensions) may be too narrow for robust layer-specific processing. 1.5d (7,680 dimensions) provides comfortable headroom. At `M/d = 1.5` and `λ = 0.43`, the portability hypothesis (Section 2.3.4) predicts moderate depth-residual reduction, consistent with the conservative sharing ratio `r = 2`. A configuration with `M/d = 1.75` and `λ ≈ 0.47` might enable `r = 3`, but the reference model opts for the safer starting point.
- **e = 0.4d** — follows from the constraint: `e = (1 − 0.43) × 3.5d / 5 = 0.4d` (2,048 dimensions, a clean power-of-2). This is not independently chosen — it is a consequence of `D`, `λ`, and `k`.
- **r = 2 (sharing ratio 1:2)** — conservative. The sharing-portability bound (Section 2.3.5) conjectures that the reference point (`M/d = 1.5`, `λ ≈ 0.43`) may support `r_max ≈ 2`. Higher ratios (1:3, 1:4) would compress expert storage further but may require greater normalized micro capacity or other changes that improve portability.
- **E = 160** — the router's output dimension. Balances pool diversity against routing complexity.
- **D = 3.5d (inter_dense = 3.5d)** — follows from the constraint. Wider than typical (vs. 2.67–3.5×d in standard architectures) but necessary to maintain uniform active compute with a meaningful micro dense-FFN.

A model targeting H200 (141 GB) with the same principles could fit more pools per exp-GPU (wider blocks, fewer subtypes), increase d_model, or allocate more KV-cache — the principles remain the same, only the point in parameter space shifts.

---

## 7. Open Questions

1. **Expert portability and the micro-capacity–sharing relationship.** The portability hypothesis (Section 2.3.4) predicts that inter-layer interference decreases as normalized micro capacity `M/d` increases, holding relevant variables fixed. Cross-layer expert substitution experiments should vary both `D/d` and `λ` so that the effects of absolute normalized width and fractional compute share can be separated. The resulting measurements should establish which secondary variables materially affect `r_max` and whether the relationship saturates.

2. **Uniform FFN budget vs. unconstrained.** Does the constraint `inter_micro + k × inter_exp = inter_dense` improve quality or systems balance relative to allowing MoE layers to have different active widths? Both validation quality and measured stage time should be reported.

3. **Micro dense-FFN sizing and the candidate `λ ≈ 0.47` ratio.** What minimum `M/d` provides sufficient layer-specific processing to make expert sharing safe? The theoretical range is 1.25–2.0, but the practical optimum may vary with `d_model`, total FFN budget, expert granularity, and sharing ratio. The recurrent `λ ≈ 0.47` examples must be treated as constructed starting points until controlled experiments show whether the ratio has any scale-stable significance.

4. **D/E pretraining allocation.** How long should D-only warm-up last, and what persistent D-batch fraction best preserves a complete dense-like path during mixed D/E pretraining without undertraining the expert pools? This balance likely changes with scale and token budget.

5. **Post-training variable-k collaboration.** What `k_C/k_E` ratio yields useful additional refinement before diminishing returns, destructive expert interaction, or grouped-GEMM inefficiency dominates? Capacity quality must be compared against an untrained higher-k inference control to isolate the value of explicit C-mode training.

6. **Hierarchical self-distillation.** Does C→E→D transfer outperform direct C→D distillation and independent regime training? Experiments should measure not only average quality but also negative transfer, calibration, long-tail recall, and whether improvements migrate into the shared micro-based path.

7. **Shared-weight gradient interference.** Which combination of regime-balanced sampling, loss weighting, gradient conflict mitigation, stopped-gradient or moving-average targets, and curriculum overlap preserves the ordering `quality_C ≥ quality_E ≥ quality_D` without weakening the best mode?

8. **Force-then-relax routing.** Which auxiliary loss and decay schedule produce broad initial exploration without permanently suppressing useful specialization? Does the optimal schedule differ between D/E pretraining and variable-k post-training?

9. **C-factor calibration.** Can `η_share` and `N_eff` predict the quality gap to matched per-layer baselines across model sizes, sharing ratios, regimes, and tasks?

10. **Router architecture and overhead.** MLP routers (`d → 512 → E`) provide richer routing decisions than linear routers but add latency. The wall-clock cost and their ability to maintain useful rankings beyond `k_E` need profiling at scale.

11. **Pipeline scheduling.** How large are pipeline bubbles under realistic mixtures of prefill, decode, variable output lengths, operating modes, and tool calls? Depth-indexed continuous batching must be evaluated under production conditions.

12. **Trunk/exp load balance.** Effective and capacity modes should be profiled separately. Attention and KV-cache may make trunk execution bandwidth-bound, while exp-GPUs depend on grouped-GEMM occupancy, routing skew, and the reduction in available virtual slots under C-mode.

13. **Scaling behavior.** The framework predicts that shared pools may become safer at larger scales because wider layer-local representations can absorb more depth-specific work. This requires controlled scaling experiments rather than parameter-count extrapolation; greater depth could also increase functional differentiation and work in the opposite direction.

14. **Cold retrieval architecture.** The extended-context mechanism is orthogonal to shared pools and should be evaluated separately.

---

## 8. Conclusion

Shared expert pools, as presented in this framework, rest on a single architectural insight: **if layer-specific processing is handled by a dedicated, always-active component (micro dense-FFN), then expert pools do not need to be layer-specific — and can be shared across layers and across replicas.**

This insight produces a coherent system:
- A **layer design** where experts refine rather than replace dense processing
- A **training architecture** where D/E pretraining builds a shared knowledge base, D→E→C post-training teaches progressively richer use of it, the micro dense-FFN remains continuously trainable, and C→E→D self-distillation transfers quality toward cheaper execution
- A **deployment architecture** where expert sharing enables 1.33–1.5 GPU per replica (under `T + q/N` accounting, with `T=1`, `q=1` for the reference design)
- A **diagnostic framework** (C-factor) that separates observable pool breadth from empirically calibrated sharing efficiency
- **Three operating regimes** (dense-like, effective, capacity) on the same hardware with the same weights

The framework is not tied to a specific model size, hardware generation, or hyperparameter configuration. The reference model (132B on H100) demonstrates one instantiation; the principles apply across the design space.

All claims are theoretical. The framework produces specific, testable predictions—about portability, learned higher-k collaboration, quality retention across D/E/C, downward self-distillation, C-factor accuracy, and deployment efficiency—that can and should be validated experimentally. The structural arguments are strong enough to warrant that investment.

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
