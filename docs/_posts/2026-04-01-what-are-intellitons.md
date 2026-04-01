---
layout: post
title: "What Are Intellitons? The Hidden Quasi-Particles Inside Large Language Models"
date: 2026-04-01
categories: [introduction, theory]
excerpt: >
  Large language models are usually described as giant statistical machines. But what if there is a
  richer, more structured story hiding inside them? This article introduces Intellitons — stable
  collective excitation patterns inside transformer residual streams — and explains why the language
  of physics may offer a surprisingly useful lens on AI internals.
---

## A new way to look at large language models

When people talk about large language models (LLMs), the standard description goes something like
this: the model reads text, compresses patterns from an enormous training corpus, and predicts the
next token. That account is correct, but it is not always satisfying. It explains *what* these
systems do, yet says very little about *how organised structure emerges inside them*.

The Intelliton project proposes a bold and visually intuitive alternative perspective. Inside the
residual stream of a transformer, there may exist relatively stable collective excitation patterns
that behave *somewhat like quasi-particles in physics*. This project gives those patterns a name:
**Intellitons**.

This is the first article in a popular science series based on the code and experimental outputs in
the [Intelliton repository](https://github.com/xiongzhp/Intelliton). Later articles will dive into
specific models and findings. This first article focuses on the idea itself: what it means, where it
comes from, and why it might be worth taking seriously.

---

## From electrons to phonons: a brief detour through physics

To understand Intellitons, it helps to understand the concept of a **quasi-particle**.

In condensed matter physics — the branch that studies solids, liquids, and materials — a
quasi-particle is *not* a fundamental entity like an electron in the Standard Model. It is a
**stable collective pattern** that emerges when many microscopic degrees of freedom act together.

A classic example is the phonon. In a crystal, billions of atoms are jostling around. Tracking each
atom individually is hopeless. But the crystal supports collective vibration waves, and those waves
behave almost like free particles: they have a well-defined energy, momentum, and even a kind of
mass. Physicists call these quantised vibrations *phonons*. They are not elementary particles, but
they are real, useful, and predictable objects for describing the crystal's behaviour.

Other examples abound:

- **Magnons** — collective spin waves in magnetic materials
- **Plasmons** — collective oscillations of electrons in a metal
- **Polarons** — electrons dressed by the lattice distortions they cause

The key insight is that a complicated substrate can support simple, emergent, trackable objects.

---

## The Intelliton idea

The Intelliton project asks whether the same logic applies to a transformer neural network.

A transformer has a **residual stream** — a vector of hidden activations that flows through each
layer and is progressively modified. At every layer, attention heads and feed-forward blocks add
their contributions to this stream. The result, layer by layer, is a transformation of the
information content.

The Intelliton idea treats this residual stream as a kind of **discrete field** living on a
token–layer lattice:

- the **token position** acts like a spatial coordinate,
- the **layer index** acts like a depth or time-like direction,
- the **hidden dimension** acts like an internal degree of freedom.

On top of this representation, the codebase in `src/` performs:

1. **Fourier analysis** over token positions to identify dominant spatial frequencies (momenta),
2. **Singular-value decomposition** over residual activations to find dominant collective modes,
3. **Propagator-based mass extraction** to measure how hard a mode is to excite or sustain,
4. **Lattice dispersion fitting** to characterise how modes propagate through the network's depth,
5. **Helicity and spin-like diagnostics** to quantify the internal complexity of each mode.

The resulting dominant modes are then catalogued as candidate **Intelliton species**.

So the working definition used in this project is deliberately pragmatic:

> **An Intelliton is a relatively stable, recurrent collective activation mode in the transformer
> residual field, identifiable across layers and prompts, and describable by effective quantities
> such as mass, momentum, spin-like complexity, helicity, and renormalisation behaviour.**

That is a scientific modelling choice, not a philosophical claim that language models literally
contain particles. Its value depends on whether it organises observations better than simpler
alternatives.

---

## Why borrow from physics?

At first glance, applying physics vocabulary to neural networks might seem like poetic
over-reaching. But there are principled reasons to try.

**Transformers are layered systems.** Information passes through many stages, each of which filters,
amplifies, and reorganises the signal. That is structurally similar to the scale transformations
studied in renormalisation group theory. It is natural to ask whether early layers are
"ultraviolet-like" (fine-grained, high-frequency) while later layers are "infrared-like"
(coarse-grained, smoothed out).

**Many physical systems share mathematical structure with transformers.** Attention is equivalent to
a certain kind of kernel smoothing. Residual connections resemble discretised differential equations.
The embedding dimension is large and acts like an internal degree of freedom. These are not arbitrary
metaphors — they are formal correspondences.

**Emergent simplicity is a real phenomenon.** Many complex systems support a small number of
dominant modes that concentrate most of the action. Turbulent fluids, financial markets, and neural
circuits all exhibit this. There is no a priori reason transformers should be different.

**A vocabulary for comparison.** If transformers do support stable collective modes, those modes give
us a language for comparing models: not just in terms of benchmark accuracy, but in terms of internal
dynamical structure. Do different model families have different "particle spectra"? Does scaling
change the mass and momentum of dominant modes? Does instruction tuning reshape the internal
excitation landscape?

---

## What does an Intelliton look like?

Each Intelliton species in the catalogue is described by a set of effective quantities:

| Quantity | Physical analogy | What it measures in the LLM |
|---|---|---|
| **Mass (pole)** | Particle pole mass | Inverse persistence of the mode through layers |
| **Mass (lattice)** | Lattice-regulated mass | Mass from dispersion relation on the token lattice |
| **Momentum** | Spatial momentum | Dominant Fourier frequency across token positions |
| **Spin-like score** | Helicity / spin | Complexity and internal structure of the mode |
| **Amplitude** | Mode strength | How strongly the mode is activated |
| **Fixed-point layer** | RG fixed point | Layer where the mode settles into stable behaviour |
| **Fixed-point type** | UV / IR / crossover | Whether the mode remains dynamic or stabilises |

These are not hand-crafted features. They are computed from data — from the actual activations of
real language models on a set of benchmark prompts.

---

## Why the word "Intelliton"?

The name combines *intelligence* and the quasi-particle suffix *-on* (as in phonon, magnon, etc.).
It signals that the object is:

- **emergent**, not elementary,
- **collective**, not individual,
- **trackable**, not just statistical,
- **connected to intelligent behaviour**, not just to random noise.

Whether Intellitons ultimately prove to be a lasting scientific concept or a useful stepping stone
depends on whether the framework continues to yield testable, reproducible, and interpretable
results.

---

## What comes next in this series

The following articles go from the abstract to the concrete:

- **Article 2** takes you inside the model `Qwen3-4B-Base` and walks through its Intelliton
  catalogue in detail — 6 species, their masses, momenta, fixed-point layers, and how they map onto
  different reasoning tasks.
- **Article 3** explores what happens when you scale from 4B to 8B parameters, and what instruction
  tuning does to the internal excitation spectrum.
- **Article 4** applies the Intelliton lens to one of the most pressing practical problems in LLM
  research: hallucination. Can internal spectral instability predict when a model is about to
  confabulate?

If the idea of quasi-particles inside language models intrigues you, stay with the series. The
results are more structured — and more surprising — than you might expect.
