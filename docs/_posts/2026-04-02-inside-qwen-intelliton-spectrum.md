---
layout: post
title: "Inside Qwen3-4B-Base: Mapping a Language Model's Internal Particle Spectrum"
date: 2026-04-02
categories: [case-study, qwen]
excerpt: >
  The clearest single demonstration of the Intelliton framework comes from Qwen3-4B-Base. This
  article walks through its complete Intelliton catalogue — 6 species with masses, momenta,
  fixed-point layers, and distinct task affinities — and explains what each finding means in plain
  language.
---

## The clearest window into Intelliton structure

Among the five models analysed in the Intelliton project, `Qwen3-4B-Base` provides the cleanest
and most interpretable results. It is a 4-billion-parameter base language model from the Qwen3
family — no instruction tuning, no alignment — which means its internal structure reflects the
organisation learned purely from unsupervised pretraining.

This article walks through every major finding for this model: the particle catalogue, momentum
structure, renormalisation group flow, and generation-time dynamics. By the end, you will have a
concrete picture of what an Intelliton catalogue looks like in practice.

---

## The pipeline: how the catalogue is built

The analysis begins with the model's **residual stream** — the hidden activation vector that flows
through every layer. For a set of diverse benchmark prompts (covering arithmetic, factual recall,
logical reasoning, pronoun tracking, and syntactic agreement), the code:

1. **Captures residual activations** at every layer for every token.
2. **Treats the activation array as a discrete field** on a token–layer lattice.
3. **Performs Fourier analysis** across token positions to extract dominant spatial modes (momenta).
4. **Applies singular-value decomposition (SVD)** layer by layer to find dominant collective modes.
5. **Fits a propagator** along the layer direction to extract an effective mass for each mode.
6. **Runs EFT / RG analysis** to identify fixed-point layers and classify modes as UV, IR, or
   crossover.
7. **Merges similar modes** across prompts to build the final species catalogue.

The result is a compact table — the Intelliton catalogue — that describes the model's dominant
internal structure with far fewer numbers than the billions of raw parameters.

![Qwen3-4B-Base particle table]({{ "/assets/images/Qwen3-4B-Base/particle_table.png" | relative_url }})
*The Intelliton catalogue for Qwen3-4B-Base. Six species, each with mass, momentum, spin-like score,
amplitude, fixed-point layer, and task affinities.*

---

## Six species, not hundreds

One of the most striking findings is how compact the catalogue is. Under this analysis pipeline,
`Qwen3-4B-Base` resolves into just **6 dominant species**: `I_0` through `I_5`.

This does not mean the model is simple. It has 4 billion parameters and a rich activation space.
But it does mean that most of the systematic, recurrent collective behaviour is concentrated in a
small number of dominant modes — much like how a complex plasma can be described mostly by its
dominant plasmon modes.

### The leading species: `I_0`

The dominant species is `I_0`, by a large margin:

| Property | Value |
|---|---|
| Amplitude | **6167.1** |
| Spin-like score | **1.84** |
| Pole mass | **0.1275** |
| Lattice mass | **0.0083** |
| Momentum | **π ≈ 3.14** |
| Fixed-point layer | **16** |
| Fixed-point type | **crossover** |
| Active tasks | arithmetic, factual recall, logical reasoning |

The amplitude of `I_0` is about 30× larger than any other species. That makes it the dominant
**backbone mode** of the model.

What does each property tell us?

**The amplitude (6167.1)** means this mode is strongly activated. When the model processes a prompt,
`I_0` is by far the most energetically prominent collective excitation.

**The spin-like score (1.84)** measures internal complexity. A value near 1 indicates a relatively
simple, coherent mode; higher values indicate a richer internal structure. `I_0` sits at a moderate
level — strong but not overly complex.

**The pole mass (0.1275)** indicates how hard the mode is to sustain through the layers. A lower
mass means the mode propagates more freely. `I_0` is relatively light, consistent with it being a
broadly present, persistent background excitation.

**The momentum at π** is perhaps the most surprising property. In the Fourier decomposition over
token positions, `I_0` peaks at the highest frequency (k ≈ π). This means it alternates strongly
between adjacent tokens — a globally "checker-board" pattern across the sequence. We return to
this below.

**The fixed-point layer (16) and type (crossover)** mean that `I_0` has not settled into a stable
infrared regime by the end of the network. It is still dynamically transitioning — remaining active
and structured all the way through.

### The secondary species: `I_1` through `I_5`

The remaining five species are much weaker in amplitude, but they are more specialised. Here is a
summary:

| Species | Top task | Momentum | Fixed-point type | Amplitude |
|---|---|---|---|---|
| `I_1` | logical reasoning | ~0 | IR | 204.3 |
| `I_2` | arithmetic | ~0 | IR | 196.1 |
| `I_3` | syntactic agreement | ~0 | IR | 188.5 |
| `I_4` | pronoun tracking | ~0 | IR | 178.2 |
| `I_5` | factual recall | ~0 | IR | 165.4 |

Several things stand out. First, each secondary species is most strongly associated with a
**different reasoning task**. This is not a hand-labelled taxonomy. The task affinities emerge
automatically from which prompts most strongly activate each mode. The fact that there is a rough
one-to-one mapping between species and task type is a structurally appealing result.

Second, all secondary species peak at **momentum near zero**. In the token-position Fourier
decomposition, k ≈ 0 corresponds to a slowly varying, globally present mode — the opposite of the
backbone mode `I_0` that alternates at k ≈ π.

Third, all secondary species are labelled **IR** (infrared), meaning they have settled into stable,
slow-varying configurations by the later layers. This contrasts with `I_0`, which remains in a
crossover regime.

---

## The split momentum structure

The coexistence of `I_0` at high momentum (k ≈ π) and `I_1`–`I_5` at low momentum (k ≈ 0) is one
of the most interesting structural features in the dataset.

In the token-lattice picture:

- **k ≈ 0 modes** are slowly varying across token positions. They represent structure that is
  relatively uniform and global across the input sequence.
- **k ≈ π modes** are maximally alternating — a kind of "anti-ferromagnetic" pattern where
  adjacent tokens have opposite phase contributions.

So the model appears to maintain *two qualitatively different types* of collective excitation
simultaneously:

1. A very strong alternating backbone mode that covers the entire sequence.
2. A set of weaker but more semantically localised modes that are task-sensitive.

Whether this split reflects something deep about how transformers organise information across token
positions remains an open question. But it is a reproducible finding in this dataset.

![Qwen3-4B-Base momentum and helicity]({{ "/assets/images/Qwen3-4B-Base/momentum_helicity_(pronoun_tracking).png" | relative_url }})
*Momentum and helicity structure for Qwen3-4B-Base on pronoun-tracking prompts.*

---

## The RG picture: where the model "settles"

The renormalisation group (RG) analysis treats the model's layers as a scale transformation. Early
layers are treated as "ultraviolet" (fine-grained, high-frequency), while later layers are
"infrared" (coarse-grained, settled).

For `Qwen3-4B-Base`, most of the secondary species (`I_1` to `I_5`) reach their fixed points
around layer **13**, while `I_0` has its fixed point at layer **16**.

In ordinary neural-network terms, one would say: the model gradually settles into stable
higher-level representations in the middle-to-late part of the stack. The secondary, specialised
modes organise themselves a little earlier; the dominant backbone mode stays active and transitional
for longer.

The EFT / RG figures below show the running mass (how each mode's effective "weight" changes with
layer), the beta function (the rate of change), and the phase diagram of transitions.

![Qwen3-4B-Base RG flow]({{ "/assets/images/Qwen3-4B-Base/rg_flow.png" | relative_url }})
*RG flow for Qwen3-4B-Base, showing how each Intelliton species evolves across the network's depth.*

![Qwen3-4B-Base EFT parameters]({{ "/assets/images/Qwen3-4B-Base/eft_parameters.png" | relative_url }})
*Effective field theory parameters extracted for Qwen3-4B-Base.*

---

## All species are "medium mass"

A curious feature of the catalogue is that **all six species are classified as medium mass**. None
are extremely light (easy to sustain) or extremely heavy (quickly decaying).

This suggests that the dominant collective modes in `Qwen3-4B-Base` occupy a relatively narrow
dynamical band. They are stable enough to recur across layers and prompts, but flexible enough to
participate in diverse tasks. The model appears to be organised around a set of moderately robust
excitations rather than a rigid, frozen structure.

![Qwen3-4B-Base mass spectrum]({{ "/assets/images/Qwen3-4B-Base/mass_spectrum_(pronoun_tracking).png" | relative_url }})
*Mass spectrum for Qwen3-4B-Base on pronoun-tracking prompts.*

---

## Dispersion relation: how modes propagate

The dispersion relation describes how the energy (or effective frequency) of a mode depends on its
momentum. In a free field theory, this is a simple parabola; in more complex systems, it can take a
variety of shapes.

For `Qwen3-4B-Base`, the dispersion relation shows the characteristic behaviour of modes on a
discrete lattice, with the dominant mode following a distinctive curve that confirms its high-momentum
nature.

![Qwen3-4B-Base dispersion relation]({{ "/assets/images/Qwen3-4B-Base/dispersion_relation.png" | relative_url }})
*Lattice dispersion relation for Qwen3-4B-Base.*

---

## Spin spectrum: internal complexity across tasks

The spin-like score measures the internal complexity of each mode — how structured its internal
degrees of freedom are. Across different task types, the spin spectrum reveals which tasks demand
more internally complex excitations.

For pronoun-tracking prompts, the spin spectrum of `Qwen3-4B-Base` shows a clear peak at `I_0`
with a moderate spin-like score, consistent with the backbone mode maintaining a moderately structured
internal configuration.

![Qwen3-4B-Base spin spectrum]({{ "/assets/images/Qwen3-4B-Base/spin_spectrum_(pronoun_tracking).png" | relative_url }})
*Spin-like score spectrum for Qwen3-4B-Base on pronoun-tracking prompts.*

---

## Phase transitions: reorganisation points in the network

The phase transition analysis identifies layers where the activation structure undergoes a
qualitative change — a point where the effective degrees of freedom reorganise.

For `Qwen3-4B-Base`, the transition map confirms that the major reorganisation occurs in the
mid-to-late layers (around layer 13–16), consistent with the RG fixed-point analysis.

![Qwen3-4B-Base phase transitions]({{ "/assets/images/Qwen3-4B-Base/phase_transitions.png" | relative_url }})
*Phase transition diagram for Qwen3-4B-Base.*

---

## What the catalogue means in plain language

Putting it all together, the `Qwen3-4B-Base` Intelliton catalogue tells a coherent story:

1. The model maintains a very strong, globally alternating backbone excitation (`I_0`) that is active
   across virtually all task types.
2. On top of that backbone, the model runs a set of five weaker, more task-specific modes that
   correspond loosely to different types of reasoning.
3. The task-specific modes organise themselves first (around layer 13), while the backbone mode
   remains active and transitional until later (around layer 16).
4. All modes are "medium mass" — stable but not rigid.
5. The dominant mode alternates at high frequency across token positions; the task-specific modes are
   more globally uniform.

This is a surprisingly structured picture for a system with 4 billion parameters. The next article
will show how this structure changes when the model is scaled up to 8B parameters or subjected to
instruction tuning.

---

## Further reading

- Continue to [Article 3: Scaling and Alignment Through the Intelliton Lens]({% post_url 2026-04-03-scaling-alignment-intellitons %})
- Continue to [Article 4: Hallucination as Internal Instability]({% post_url 2026-04-04-intellitons-and-hallucination %})
- Return to [Article 1: What Are Intellitons?]({% post_url 2026-04-01-what-are-intellitons %})
