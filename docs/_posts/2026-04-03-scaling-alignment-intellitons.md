---
layout: post
title: "Scaling and Alignment Through the Intelliton Lens"
date: 2026-04-03
categories: [comparison, scaling, alignment]
excerpt: >
  What happens to a model's internal quasi-particle spectrum when you double the parameter count?
  What does instruction tuning do to the excitation landscape? This article compares four Qwen3
  models — 4B vs 8B, Base vs Instruct — and adds Mistral-7B-v0.3 for a cross-family perspective.
---

## Two of the biggest questions in LLM science

Two of the most discussed phenomena in large language model research are **scaling** and
**alignment**. Scaling means training bigger models; alignment (here: instruction tuning) means
fine-tuning a model to follow instructions and behave helpfully.

Both interventions are known to improve benchmark performance. But do they change the *internal
structure* of the model? Do they reshape the quasi-particle spectrum?

The Intelliton analysis of five models — `Qwen3-4B-Base`, `Qwen3-4B`, `Qwen3-8B-Base`, `Qwen3-8B`,
and `Mistral-7B-v0.3` — provides a concrete, data-driven answer.

---

## Base versus Instruct: what alignment does

The clearest comparison is between a base model and its instruction-tuned counterpart in the same
family.

### Qwen3-4B-Base vs. Qwen3-4B

| Property | Qwen3-4B-Base | Qwen3-4B |
|---|---|---|
| Number of species | **6** | **5** |
| `I_0` amplitude | 6167.1 | 6562.4 |
| Dominant momentum | **k ≈ π** | **k ≈ 1.885** |
| Secondary momenta | k ≈ 0 | k ≈ 1.885 (shared) |
| Fixed-point types | IR + crossover | all **crossover** |
| Grounded profile mean | 32.68 | 29.13 |

The most striking difference is in **momentum structure**.

In `Qwen3-4B-Base`, the dominant mode `I_0` peaks at k ≈ π (high frequency, alternating pattern),
while the five secondary modes all peak near k ≈ 0 (low frequency, global pattern). There is a
clean split.

In `Qwen3-4B` (the instruction-tuned version), the dominant mode shifts to k ≈ 1.885, and the
secondary modes also cluster around the same momentum. The model becomes more **homogeneous** in
its momentum structure. The clean split between backbone and task-specific modes disappears.

The fixed-point type change is equally telling. In the base model, five of the six species are
labelled **IR** (settled, stable), while `I_0` is **crossover** (still transitioning). In the
instruct model, **all** species are labelled **crossover**. Alignment appears to push the model's
collective modes into a more uniformly active, less settled dynamical regime.

One possible interpretation:

> **Instruction tuning compresses or reorganises the internal excitation landscape into a more
> uniform effective regime. The spectral diversity that exists in the base model is partially
> smoothed out, and more modes are kept in a dynamically transitional state.**

This does not mean instruction tuning is worse. It may mean the model's degrees of freedom are being
regularised toward instruction-following behaviour, possibly at the cost of some internal
differentiation.

![Qwen3-4B particle table]({{ "/assets/images/Qwen3-4B/particle_table.png" | relative_url }})
*The Intelliton catalogue for Qwen3-4B (instruction-tuned). Compare with Qwen3-4B-Base to see
the homogenisation of momentum structure.*

---

## Scaling from 4B to 8B: what more parameters do

The next comparison holds model family (Qwen3) and training type (base) constant and varies the
parameter count.

### Qwen3-4B-Base vs. Qwen3-8B-Base

| Property | Qwen3-4B-Base | Qwen3-8B-Base |
|---|---|---|
| Number of species | 6 | **7** |
| `I_0` amplitude | 6167.1 | **7908.4** |
| Dominant momentum | k ≈ π | k ≈ π |
| Grounded profile mean | 32.68 | **60.26** |
| Grounded–hallucination separation | moderate | **larger** |

The 8B model has one more species. Its leading mode `I_0` is **28% stronger** in amplitude.
Most strikingly, the grounded generation profile mean nearly **doubles** (32.68 → 60.26).

In Intelliton terms, scaling up does not simply add more parameters uniformly. It appears to
**amplify the dominant dynamical sectors** of the model, making the leading quasi-particle modes
substantially stronger and the grounded generation trajectory more sharply defined.

The momentum structure is also richer in the 8B model. While the leading mode still peaks at
k ≈ π, many of the secondary species cluster around k ≈ 1.885 rather than strictly k ≈ 0. This
suggests that intermediate-scale spatial organisation becomes more visible as the model grows.

![Qwen3-8B-Base particle table]({{ "/assets/images/Qwen3-8B-Base/particle_table.png" | relative_url }})
*The Intelliton catalogue for Qwen3-8B-Base. The leading mode is stronger, and the species set
is slightly larger compared with Qwen3-4B-Base.*

---

## Scaling plus alignment: Qwen3-8B

When both scaling and instruction tuning are applied — `Qwen3-8B` — the results follow the pattern
suggested by the two effects separately.

| Property | Qwen3-8B-Base | Qwen3-8B |
|---|---|---|
| Number of species | 7 | **6** |
| `I_0` amplitude | 7908.4 | ~7600 |
| Grounded profile mean | 60.26 | **57.14** |
| Fixed-point types | mixed | more crossover |

Instruction tuning at 8B scale slightly reduces the species count (7 → 6) and slightly lowers the
dominant mode amplitude and grounded profile mean, consistent with the homogenisation effect seen
at 4B scale. But the 8B instruct model remains far stronger than the 4B models in its grounded
trajectory profile.

The combined takeaway is clean:

- **Scaling to 8B** increases the strength of dominant collective modes and sharpens the grounded
  generation signal.
- **Instruction tuning** slightly compresses or regularises that internal structure, reducing
  species count and reducing the grounded/hallucination separation.

These two effects appear to be largely independent and roughly additive in their impact on the
Intelliton spectrum.

![Qwen3-8B particle table]({{ "/assets/images/Qwen3-8B/particle_table.png" | relative_url }})
*The Intelliton catalogue for Qwen3-8B (instruction-tuned 8B model).*

---

## A completely different family: Mistral-7B-v0.3

The most dramatic contrast in the entire comparison set comes from `Mistral-7B-v0.3`, a 7B model
from a different architecture family.

| Property | Qwen3-4B-Base | Mistral-7B-v0.3 |
|---|---|---|
| Number of species | 6 | **25** |
| `I_0` amplitude | 6167.1 | **249.4** |
| Grounded profile mean | 32.68 | **21.48** |
| Grounded profile std | moderate | **46.56** (very large) |
| Fixed-point types | IR + crossover | UV + IR + crossover |

Under the same analysis pipeline, Mistral produces **25 species** — more than four times as many as
any Qwen model. This is a striking result.

The leading species `I_0` in Mistral has an amplitude of only 249.4, compared with 6167 in
`Qwen3-4B-Base`. In other words, the Mistral model does not have a strongly dominant backbone
excitation. Its collective mode landscape is much more **fragmented**: many modes of comparable
strength, rather than one overwhelming mode with several weak followers.

Mistral also shows **UV-labelled species** — modes that remain in a fine-grained, ultraviolet-like
dynamical state throughout the network, rather than flowing toward infrared stability. This suggests
a more persistent fine-grained structure in Mistral's layers compared with Qwen.

The generation dynamics also differ. The grounded profile mean for Mistral (21.48) is lower than
for all Qwen models, and the standard deviation (46.56) is much larger. The Intelliton metric,
calibrated using Qwen, describes Mistral's trajectory space as much noisier and more turbulent.

One interpretation:

> **Qwen organises its internal computation around a few very strong collective modes. Mistral
> spreads the load more broadly across many smaller modes. Under this analysis, they have genuinely
> different "particle physics" inside.**

Whether this difference reflects architectural choices, training data, training procedure, or some
combination is an open question. But the Intelliton framework makes the difference visible and
measurable.

![Mistral-7B-v0.3 particle table]({{ "/assets/images/Mistral-7B-v0.3/particle_table.png" | relative_url }})
*The Intelliton catalogue for Mistral-7B-v0.3. 25 species, a far more fragmented landscape than any
Qwen model.*

---

## A summary table

| Model | Species | `I_0` Amplitude | Momentum structure | Fixed-point types | Grounded mean |
|---|---|---|---|---|---|
| Qwen3-4B-Base | 6 | 6167 | π + 0 (split) | IR + crossover | 32.68 |
| Qwen3-4B | 5 | 6562 | 1.885 (homogeneous) | all crossover | 29.13 |
| Qwen3-8B-Base | 7 | 7908 | π + 1.885 (richer) | mixed | 60.26 |
| Qwen3-8B | 6 | ~7600 | 1.885 (homogeneous) | more crossover | 57.14 |
| Mistral-7B-v0.3 | 25 | 249 | fragmented | UV + IR + crossover | 21.48 |

---

## Conclusions

The Intelliton comparison across these five models yields several clear empirical regularities:

1. **Instruction tuning homogenises the momentum structure** and shifts more species into crossover
   regimes, reducing internal spectral diversity.
2. **Scaling from 4B to 8B strengthens the dominant dynamical sectors**, producing a more strongly
   occupied and more sharply separated Intelliton landscape.
3. **Different model families can have qualitatively different internal spectra** — Qwen is dominated
   by a few strong modes, Mistral is more fragmented and UV-rich.
4. The Intelliton framework provides a vocabulary for these differences that goes beyond benchmark
   accuracy or parameter count alone.

The next article turns to one of the most practically important applications of this framework:
using the Intelliton lens to study **hallucination** — and asking whether internal spectral
instability can be a diagnostic signal for when a model is about to confabulate.

---

## Further reading

- Continue to [Article 4: Hallucination as Internal Instability]({% post_url 2026-04-04-intellitons-and-hallucination %})
- Return to [Article 2: Inside Qwen3-4B-Base]({% post_url 2026-04-02-inside-qwen-intelliton-spectrum %})
- Return to [Article 1: What Are Intellitons?]({% post_url 2026-04-01-what-are-intellitons %})
