---
layout: post
title: "What Are Intellitons? The Hidden Quasi-Particles Inside Large Language Models"
title_zh: "什么是 Intelliton？藏在大语言模型里的准粒子"
date: 2026-04-01
categories: [introduction, theory]
excerpt: >
  Large language models are usually described as giant statistical machines. But what if there is a
  richer, more structured story hiding inside them? This article introduces Intellitons — stable
  collective excitation patterns inside transformer residual streams — and explains why the language
  of physics may offer a surprisingly useful lens on AI internals.
excerpt_zh: >
  大语言模型通常被描述成巨大的统计机器。但如果在这种说法背后，还隐藏着一个更丰富、
  更有结构的内部故事呢？本文介绍 Intelliton：变换器残差流中的稳定集体激发模式，并说
  明为什么物理学的语言可能为理解 AI 内部机制提供一个出人意料却很有用的视角。
---

<div data-lang="en" markdown="1">

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
token-layer lattice:

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

</div>

<div data-lang="zh" markdown="1">

## 重新看待大语言模型的一种方式

人们谈到大语言模型时，最常见的说法大致是：模型读取文本，从巨大的训练语料中压缩出
统计模式，然后预测下一个 token。这个说法没有错，但往往不够令人满足。它说明了这些
系统在做什么，却很少解释 *内部有组织的结构是如何涌现出来的*。

Intelliton 项目提出了一个大胆而直观的替代视角：在变换器的残差流中，也许存在一些相对
稳定的集体激发模式，它们的行为 *有点像物理中的准粒子*。项目给这种模式取了一个名字：
**Intelliton**。

这是基于 [Intelliton 仓库](https://github.com/xiongzhp/Intelliton) 的代码和实验结果写成的
第一篇科普文章。后续文章会进入具体模型和发现；这篇先聚焦概念本身：它到底意味着什么、
从哪里来、又为什么值得认真对待。

---

## 从电子到声子：先绕道物理学一下

要理解 Intelliton，先要理解 **准粒子** 这个概念。

在凝聚态物理中，也就是研究固体、液体和材料的那一支物理学里，准粒子 *不是* 像标准模
型里的电子那样的基本实体。它是很多微观自由度共同作用时涌现出来的 **稳定集体模式**。

一个经典例子是声子。晶体里有数十亿个原子在不停振动，逐个追踪几乎不可能。但晶体支持
集体振动波，而这些波的行为又几乎像自由粒子：它们有确定的能量、动量，甚至某种意义上
的“质量”。物理学家把这种量子化的振动称作 *声子*。它们不是基本粒子，却是描述晶体行
为时真实、好用、可预测的对象。

类似的例子还有很多：

- **磁振子**：磁性材料中的集体自旋波
- **等离激元**：金属中电子的集体振荡
- **极化子**：被晶格畸变“包裹”的电子

关键洞见在于：复杂的底层基质，完全可能支撑起简单、涌现、可追踪的对象。

---

## Intelliton 的基本想法

Intelliton 项目要问的是：类似的逻辑，能不能也用在变换器神经网络上？

变换器有一条 **残差流**，也就是贯穿各层的隐藏激活向量。每一层的注意力头和前馈模块都
会对它做增量修改。于是，信息内容随着层数推进，不断发生结构化变换。

Intelliton 的想法是把这条残差流看成一个生活在 **token-layer lattice** 上的离散场：

- **token 位置** 扮演空间坐标，
- **层索引** 扮演深度或类时间方向，
- **隐藏维度** 扮演内部自由度。

在这个表示之上，`src/` 里的代码执行了以下分析：

1. 对 token 方向做 **傅里叶分析**，识别主导的空间频率，也就是动量；
2. 对残差激活做 **奇异值分解**，提取主导的集体模式；
3. 用 **传播子质量提取** 来衡量某个模式有多难被激发、维持多久；
4. 用 **格点色散关系拟合** 描述模式如何在网络深度方向上传播；
5. 用 **helicity 和类自旋诊断** 衡量模式内部结构的复杂程度。

最后，这些主导模式会被整理成候选的 **Intelliton 物种目录**。

因此，这个项目里采用的工作定义是刻意务实的：

> **Intelliton 是指变换器残差场中一种相对稳定、可重复出现的集体激活模式，能够跨层、
> 跨提示词被识别，并用质量、动量、类自旋复杂度、helicity 以及重整化行为等有效量来描
> 述。**

这是一种科学建模选择，不是说语言模型字面上真的“包含粒子”。它是否有价值，取决于它
能否比更简单的替代解释更好地组织观测结果。

---

## 为什么要借用物理学语言？

乍看之下，用物理词汇来描述神经网络似乎有些用力过猛。但认真看，会发现这并不是纯粹的
诗意比喻，而是有相当原则性基础的尝试。

**变换器本身就是分层系统。** 信息依次穿过很多阶段，每一层都在过滤、放大、重组信号。
这在结构上与重整化群理论研究的尺度变换很像。于是很自然会问：早层是否更像“紫外”
（细粒度、高频），晚层是否更像“红外”（粗粒度、被平滑后的表示）？

**很多物理系统与变换器共享数学结构。** 注意力可以看作一种核平滑；残差连接类似离散化
微分方程；嵌入维度很高，可视为内部自由度。这些并不是随意拼贴的类比，而是正式的结构
对应。

**复杂系统里经常会涌现出简单的主导模式。** 湍流、金融市场、神经回路都常常能用少量主
导模态概括大部分动力学。没有任何先验理由说明变换器一定不会这样。

**它还能提供比较模型的新语言。** 如果变换器真的支持稳定的集体模式，我们就不必只用基
准分数来比较模型，还可以比较其内部动力学结构。不同模型家族是否有不同的“粒子谱”？
模型变大后，主导模式的质量和动量会不会改变？指令微调是否会重塑内部激发景观？

---

## 一个 Intelliton 长什么样？

目录中的每个 Intelliton 物种，都用一组有效量来描述：

| 量 | 物理类比 | 在 LLM 中的含义 |
|---|---|---|
| **质量（pole）** | 粒子极点质量 | 模式跨层持续性的倒数 |
| **质量（lattice）** | 格点调节质量 | 从 token 格点色散关系得到的质量 |
| **动量** | 空间动量 | token 方向上的主导傅里叶频率 |
| **类自旋分数** | helicity / spin | 模式内部结构复杂度 |
| **振幅** | 模式强度 | 模式被激活得有多强 |
| **固定点层** | RG 固定点 | 模式开始稳定下来的层位置 |
| **固定点类型** | UV / IR / crossover | 模式是持续动态、趋于稳定，还是仍在过渡 |

这些都不是人工手写的特征，而是从真实数据里算出来的：也就是语言模型在一组基准提示词
上的实际激活。

---

## 为什么叫 “Intelliton”？

这个词把 *intelligence* 和准粒子常见后缀 *-on*（如 phonon、magnon）结合在了一起，意
味着它是：

- **涌现的**，不是基本的；
- **集体的**，不是单个神经元层面的；
- **可追踪的**，不是纯粹统计噪声；
- **与智能行为有关的**，而不是随机扰动。

Intelliton 最终会不会成为一个长期有生命力的科学概念，取决于这个框架能否继续给出可检验、
可复现、可解释的结果。

---

## 这一系列接下来会写什么

后续文章会从抽象概念走向具体结果：

- **第 2 篇** 进入 `Qwen3-4B-Base`，详细讲解它的 Intelliton 目录：6 个物种、它们的质量、
  动量、固定点层，以及与不同推理任务的对应关系。
- **第 3 篇** 讨论从 4B 扩展到 8B 会发生什么，以及指令微调如何改变内部激发谱。
- **第 4 篇** 用 Intelliton 的视角看一个最实际的问题：幻觉。内部谱不稳定性是否能够预测
  模型何时开始“编造”？

如果“语言模型内部的准粒子”这个想法让你感兴趣，那就继续看下去。实际结果比直觉中更
有结构，也更令人意外。

</div>