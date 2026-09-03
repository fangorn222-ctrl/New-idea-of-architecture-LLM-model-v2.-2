# # Shared Expert Pools: An Architectural Framework for Efficient MoE Design and Deployment

**Version 2.4 — September 2026**

A phase-local expert-sharing and serving framework for trillion-scale MoE language models.

The core premise has independent empirical support from [UniPool (arXiv:2605.06665)](https://arxiv.org/abs/2605.06665), which demonstrates that shared expert pools match or outperform standard per-layer MoE baselines. This framework extends the principle toward deployment-oriented, trillion-scale design.

---

## What This Proposes

Instead of giving every MoE layer its own independent expert pool, adjacent layers share a common pool — while keeping their own attention, layer-specific dense processing (micro dense-FFN), and routers. The micro dense-FFN absorbs all depth-specific work *before* experts are invoked, so experts perform only generic refinement that is inherently portable across layers.

**Key elements:**

- **Sequential micro dense-FFN → routed experts** — experts never do layer-specific work, making sharing architecturally natural
- **Parametric design space {D, λ, k, r}** — four parameters define all valid configurations; expert width is a derived quantity, not a free parameter
- **Testable portability hypothesis** — inter-layer interference should decrease monotonically with micro dense-FFN capacity (∂D_l^residual/∂M < 0)
- **Three operating regimes (D/E/C)** from the same weights — dense-like (trunk-only), effective (standard MoE), and capacity (extended top-k)
- **Trunk/expert GPU disaggregation** — 1–1.5 GPU per logical replica regardless of nominal model size
- **C-factor quality metric** — separates observable pool utilization from empirically calibrated sharing efficiency

A 132B reference model on 8×H100 demonstrates one instantiation of the framework.

## What This Does NOT Provide

No experimental validation. No training runs. No benchmark results. All claims are theoretical — structural arguments, parameter-count arithmetic, and hardware deployment analysis. The framework produces testable predictions that require empirical verification.

## Version History

This repository tracks the evolution of the idea:

| Version | Date | Description |
|---|---|---|
| [v1.0](shared_expert_pools.md) | June 2026 | Initial proposal — single 1.15T model, deployment economics, four quality-preservation mechanisms |
| [v2.2](shared_expert_pools_v2_2.md) | August 2026 | Framework rewrite — principles vs. parameters, sequential processing insight, training recipe, C-factor, 132B reference model on H100 |
| **[v2.4](shared_expert_pools_v2_4.md)** | September 2026 | Current version — parametric formulation {D, λ, k, r}, portability hypothesis, composable-primitives semantics, D/E/C training architecture with hierarchical self-distillation, variable-k nested ranking |

## Author

**Aleksei Ketslakh** — neurologist (Kaplan Medical Center, Israel). This work was developed independently as a theoretical exploration, without institutional affiliation to any AI lab or access to training infrastructure.

## Acknowledgements

I would like to thank the AI systems that served as intellectual collaborators throughout this work:

- **Opus 4.6** (Anthropic) — extended architectural discussions, hardware deployment analysis, identification of the UniPool paper as independent validation, document assembly and structuring
- **ChatGPT 5.6 sol** (OpenAI) — parametric formalization of the design space {D, λ, k, r}, basis-decomposition formulation of expert semantics (F(x,l) ≈ Micro_l(x) + Σ g_i · E_i), the portability hypothesis formulation (∂D_l^residual/∂M < 0), storage-compute tradeoff analysis, and substantial contributions to the D/E/C training architecture including hierarchical self-distillation
- **Grok, Gemini, Copilot, Kimi, DeepSeek** — cross-validation of architectural reasoning and parameter-count arithmetic

Each system contributed domain knowledge, mathematical formalization, and critical review that shaped the framework. The collaboration itself — a physician developing an AI architecture framework through extended dialogue with multiple AI systems — may be of independent interest as a case study in AI-assisted theoretical research.

All architectural claims, assumptions, mistakes, and final decisions remain my own responsibility.

---

*No experimental validation has been performed. This is a testable architectural hypothesis.*
