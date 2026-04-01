---
layout: post
title: "Hallucination as Internal Instability: An Intelliton Perspective"
date: 2026-04-04
categories: [hallucination, applications]
excerpt: >
  Hallucination — when a language model confidently produces false or unsupported information — is
  one of the most pressing practical problems in LLM research. This article explores what the
  Intelliton framework reveals about hallucination: not as an output-level mistake, but as an
  instability of internal collective modes during generation.
---

## Beyond "wrong output"

When a language model hallucinates, the surface-level observation is simple: it produces text that
is incorrect, unsupported, or fabricated. But this description raises a deeper question: what is
*happening inside the model* when it hallucinates?

One common intuition is that hallucination is random — a kind of noise or statistical accident in
the token prediction process. Another is that it reflects gaps in training data. Both of these
accounts may be partially right, but they are not mechanistic: they do not tell us *where* in the
model the failure originates, or whether it corresponds to a detectable internal signal.

The Intelliton framework offers a different angle. Instead of treating hallucination as a property
of the output, it treats it as a property of the **internal dynamical trajectory** during generation.

The central hypothesis is this:

> **Hallucination may correspond to a regime of weaker, less coherent, and more fragmented
> Intelliton activity — a trajectory that stays farther from the "grounded sector" of the model's
> internal quasi-particle space.**

This article explains the evidence for that hypothesis, focusing on `Qwen3-4B-Base` as the primary
example.

---

## How hallucination is studied in the Intelliton framework

The module `src/hallucination_diagnostic.py` compares two types of prompts:

- **Grounded prompts**: questions that have factual, verifiable answers the model has likely
  encountered in training (e.g. "What is the capital of France?").
- **Hallucination-prone prompts**: questions designed to invite confabulation — factoid-sounding
  questions about obscure, ambiguous, or partially fabricated information that the model is likely
  to "fill in" plausibly but incorrectly.

For each prompt type, the analysis computes several internal metrics:

| Metric | What it measures |
|---|---|
| **Singular-value spectrum divergence** | How different the activation modes are from the grounded baseline |
| **Coherence** | How concentrated and stable the dominant singular modes are |
| **Mode stability** | Whether the dominant species remains consistent across generation steps |
| **Entropy gap** | How spread out the energy is across modes |
| **Critical layers** | Which layers show the largest divergence from grounded behaviour |

These metrics are computed *during generation* — step by step, as the model produces each new
token — not just at the final output.

![Qwen3-4B-Base hallucination diagnostics]({{ "/assets/images/Qwen3-4B-Base/hallucination_diagnostics.png" | relative_url }})
*Hallucination diagnostics for Qwen3-4B-Base. The figure compares spectral signatures between
grounded and hallucination-prone prompts.*

---

## The trajectory evidence

The generation-time trajectory data provides the clearest picture. The file
`intelliton_trajectory_summary.csv` for `Qwen3-4B-Base` records, for each generation step and
each prompt type, the mean mode activation shift and the grounded deviation.

### Grounded prompts: rising and coherent

For grounded factual prompts, the mean mode activation shift starts around **1.10** and rises to
about **1.37** over the first 8 generation steps. Top species occupation also rises, from about
**70.1% to 74.1%**.

This means that as the model commits to a factual answer, the dominant Intelliton sector becomes
**stronger and more organised**. The model is moving toward a more concentrated, coherent internal
state.

### Hallucination-prone prompts: weak and diverging

For hallucination-prone prompts, the picture is strikingly different.

The mean mode activation shift stays at only **0.32–0.41** throughout generation — roughly one
third of the grounded value. The grounded deviation remains strongly negative (roughly **-9 to -11**
across most generation steps).

In plain language: hallucination-prone generation produces an internal trajectory that is both
**weaker in overall activation** and **farther from the grounded sector of Intelliton space**.

The hallucination case is not just "wrong output at the end." It is a persistently different
internal state throughout the generation process.

### Style prompts: the intermediate case

Stylistic continuation prompts — prompts asking the model to continue a piece of creative writing
without strong factual constraints — occupy an intermediate position. Their activation shift is
higher than hallucination-prone prompts but lower than grounded factual prompts.

This is a meaningful calibration check: style generation is not simply failure, but it is also not
anchored to factual grounding. The Intelliton metric places it appropriately between the two
extremes.

![Qwen3-4B-Base Intelliton trajectory]({{ "/assets/images/Qwen3-4B-Base/intelliton_trajectory_merged.png" | relative_url }})
*Generation-time Intelliton trajectories for Qwen3-4B-Base. Grounded (top), style (middle), and
hallucination-prone (bottom) prompts show qualitatively different internal dynamical profiles.*

---

## Transition graphs: which species dominate stable generation

The Intelliton transition graph shows which species transitions are most common during generation,
and how strong those transitions are in terms of mode activation.

For **grounded prompts** in `Qwen3-4B-Base`, the dominant self-transitions are:

| Transition | Count | Mean target activation shift |
|---|---|---|
| `I_5 → I_5` | **110** | (strong) |
| `I_1 → I_1` | 13 | (moderate) |
| `I_2 → I_2` | 6 | (moderate) |

The generation stays largely within `I_5` — the factual-recall species — with occasional excursions
into `I_1` (logical reasoning) and `I_2` (arithmetic).

For **hallucination-prone prompts**, `I_5 → I_5` remains the most common transition, but its mean
target activation shift is much smaller. There is also more mixing among `I_1`, `I_3`, and `I_5`,
suggesting that the internal trajectory becomes more fragmented and less dominated by any single
species.

![Qwen3-4B-Base transition graph]({{ "/assets/images/Qwen3-4B-Base/intelliton_transition_graph.png" | relative_url }})
*Species transition graph for Qwen3-4B-Base. Grounded generation is dominated by strong self-loops;
hallucination shows a weaker, more mixed pattern.*

---

## A multiple-failure-mode picture of hallucination

One of the most useful aspects of the Intelliton framework is that it suggests hallucination is not
necessarily a single phenomenon. The trajectory data is consistent with at least four distinct
failure modes:

1. **Grounded excitation decay**: the leading Intelliton for factual tasks (here `I_5`) fails to
   maintain its activation, causing the model to "lose grip" on the factual sector.

2. **Species fragmentation**: instead of a dominant self-loop, the trajectory becomes a mixture of
   several species without a clear attractor.

3. **Spectral broadening**: the singular-value spectrum spreads out, indicating a loss of coherence
   in the dominant collective modes.

4. **Distance from grounded baseline**: the trajectory drifts away from the region of Intelliton
   space that characterises correct factual generation.

Whether a given hallucination episode involves one or all four of these failure modes may depend on
the specific model and the type of hallucination. But the framework gives us language and metrics
for distinguishing them.

---

## Implications and future directions

If this picture holds up under further investigation, it opens several practical possibilities.

### Early warning signals

Since the deviation from grounded Intelliton trajectories is detectable at the very first generation
steps, it is in principle possible to flag potential hallucinations *before* the full output is
produced. This could be the basis for hallucination early warning systems.

### Intervention on unstable species

If a particular species is identified as responsible for grounded, factual generation, it may be
possible to stabilise or amplify that species during inference using model steering techniques. The
codebase already includes modules such as `src/gauge_intervention.py` that hint at this direction.

### Prompt strategies for grounded generation

If certain prompts consistently lead to strong grounded Intelliton trajectories, understanding their
structure could inform better prompting strategies — ways to keep the model inside the grounded
sector of its internal space.

### Model comparison by internal stability

The Intelliton hallucination metric provides a new axis for comparing models — not just by accuracy
on a benchmark, but by the robustness and coherence of their internal factual-grounding sector. A
model with a stronger, more stable `I_5`-like species may be inherently more reliable for factual
tasks.

---

## Caveats and open questions

As with all findings in this project, several important caveats apply.

**The hallucination-prone prompts are designed, not naturally occurring.** The distinction between
"grounded" and "hallucination-prone" is imposed by the prompt design. In real-world use, the
boundary is less clear.

**Correlation is not causation.** The Intelliton trajectory differences are *associated* with
hallucination-prone prompts, but it has not yet been established that fixing the trajectory would
prevent hallucination.

**The pipeline has design choices.** Different prompt sets, sequence lengths, and analysis
parameters would produce different catalogs and possibly different conclusions.

**The metric is relative, not absolute.** The "grounded deviation" is measured relative to a
baseline grounded trajectory. Its meaning depends on the quality of that baseline.

Despite these caveats, the structural pattern — grounded generation being internally stronger,
more coherent, and closer to a well-defined attractor — is consistent across all generation steps
and both task splits examined in `Qwen3-4B-Base`.

---

## Where this series has taken us

This final article completes the four-part popular science series on Intellitons:

1. **[What Are Intellitons?]({% post_url 2026-04-01-what-are-intellitons %})** — The quasi-particle
   idea and why it might apply to transformers.
2. **[Inside Qwen3-4B-Base]({% post_url 2026-04-02-inside-qwen-intelliton-spectrum %})** — A
   detailed walkthrough of a model's complete Intelliton catalogue.
3. **[Scaling and Alignment]({% post_url 2026-04-03-scaling-alignment-intellitons %})** — How
   parameter count and instruction tuning reshape the internal excitation spectrum.
4. **Hallucination as Internal Instability** (this article) — Hallucination as a detectable
   internal dynamical regime.

The Intelliton framework is a young and exploratory research programme. But its outputs are concrete,
its comparisons are reproducible, and its language is — at least arguably — more informative than
treating language models as opaque statistical engines.

The goal is to develop a vocabulary that makes the internal life of neural networks legible. Whether
Intellitons are ultimately the right vocabulary remains to be seen. But the evidence so far suggests
they are pointing at something real.

---

## Further reading

- Explore the full codebase at [github.com/xiongzhp/Intelliton](https://github.com/xiongzhp/Intelliton)
- Read the accompanying technical paper (`intelliton_arxiv_paper.pdf`) for the formal analysis
- Return to [Article 1: What Are Intellitons?]({% post_url 2026-04-01-what-are-intellitons %})
