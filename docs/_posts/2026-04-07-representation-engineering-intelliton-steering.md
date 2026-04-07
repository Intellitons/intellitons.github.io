---
layout: post
title: "Representation Engineering and Intelliton Steering: A Research Proposal"
title_zh: "表征工程与 Intelliton 引导：一份研究提案"
date: 2026-04-07
categories: [representation-engineering, alignment, research-directions]
excerpt: >
  Representation engineering intervenes directly on a model's internal activations to steer its
  behaviour — without fine-tuning. The Intelliton framework provides a natural language for
  describing those interventions: they are changes to specific Intelliton species. This article
  proposes a research direction that turns the Intelliton species catalogue into a steering map.
excerpt_zh: >
  表征工程在不做微调的前提下，直接对模型的内部激活进行干预，从而引导模型行为。Intelliton
  框架为描述这些干预提供了一套自然语言：它们就是对特定 Intelliton 物种的改变。本文提出一个
  研究方向，将 Intelliton 物种目录变成一张可操作的引导地图。
---

<div data-lang="en" markdown="1">

## Two ideas that belong together

**Representation engineering** is the practice of directly reading and writing to a model's
internal activations — without changing any weights — to steer its behaviour. A growing body of
work shows that concepts like "happiness", "authority", "political bias", and "honesty" can be
encoded as linear directions in the residual stream, and that adding or subtracting a small multiple
of those directions at inference time reliably changes what the model outputs.

**The Intelliton framework** characterises the residual stream as a space of quasi-particle-like
modes. It extracts recurring patterns and labels them by their spectral properties: momentum,
spin-like complexity, mass, and helicity.

These two ideas are describing the same object — the residual stream — at different levels of
abstraction. Representation engineering says "this direction steers this behaviour". The Intelliton
framework says "this mode has these spectral properties". Combining them makes both more useful.

---

## What representation engineering can do (and what it cannot say on its own)

The tools most associated with representation engineering — activation addition, contrastive
activation analysis, and the ARA method used to jailbreak Gemma 4 — share a common limitation:
they can identify *which direction to push* but they say little about *the structure of that
direction in the broader activation space*.

Specifically:

- **Activation addition** adds a fixed direction to a chosen layer's residual stream at every
  token position. It works reliably for simple concepts, but it can degrade performance when the
  steered direction overlaps with important task-solving modes.

- **Contrastive activation analysis** (the core of Abliteration) identifies the mean difference
  between two contrastive sets of activations. It finds the refusal direction efficiently, but it
  does not tell you how many modes are involved, what those modes' propagation properties are, or
  how much overlap they have with the task-solving modes you want to preserve.

- **ARA** improves on simple subtraction by working in a low-rank subspace rather than a single
  direction. It uses SVD to separate the refusal subspace, but it does not connect the separated
  components to a broader characterisation of the model's mode landscape.

The Intelliton framework fills exactly these gaps.

---

## The steering map proposal

The research direction proposed here is to build what we can call an **Intelliton steering map**:
a catalogue that annotates each Intelliton species with its likely behavioural role, its
approximate layer range, its rank in the residual stream, and its overlap with other species.

### Building the map: three ingredients

**Ingredient 1 — Task probes**

Use the five prompt families from `src/datasets.py` (pronoun tracking, factual recall, logical
reasoning, arithmetic, syntactic agreement) to establish which Intelliton species are activated by
which kind of task. This is already partially done by the existing analysis.

**Ingredient 2 — Behavioural probes**

Add a new class of probes targeting RLHF-trained behaviours:
- refusal (harmful vs. harmless prompts),
- sycophancy (flattery vs. neutral prompts),
- political neutrality (controversial vs. neutral framings),
- verbosity control (instructed-brief vs. instructed-elaborate prompts).

Run the same Intelliton analysis on each behavioural probe set and record which species respond.

**Ingredient 3 — Cross-probe overlap**

Compute the pairwise cosine similarity between all per-layer refusal vectors and all per-layer
task-activation vectors. Species with low overlap across all task probes are good steering targets:
adding or removing them will not bleed into task performance.

---

## The connection to ARA

The ARA technique constructs a rank-$$k$$ penalty matrix $$\Delta W$$ that projects out the
refusal subspace from the model's weight matrices. In Intelliton terms, $$\Delta W$$ is a
targeted suppression of a small set of Intelliton species.

The key claim of ARA is that a higher-rank intervention is safer than a rank-1 intervention
(simple vector subtraction) because the refusal behaviour in a capable reasoning model spans
multiple entangled modes. If you only remove the rank-1 component, the remaining components
continue to generate partial refusals or degrade the model's reasoning.

This claim can be tested directly using the Intelliton framework:

1. Compute the Intelliton spectrum of an instruction-tuned model on harmful prompts.
2. Identify the modes that are most active on harmful prompts and least active on harmless prompts.
3. Measure whether those modes are clustered in a low-dimensional subspace of the per-layer SVD
   basis, or whether they are spread across many independent directions.

If they are clustered, ARA's rank-$$k$$ approach is justified and the clustering rank $$k$$ can be
estimated from the Intelliton spectrum before any jailbreak attempt is made. If they are spread,
simple subtraction methods are expected to leave residual refusal capability or cause broader
collateral damage.

---

## A practical application: zero-shot concept injection

The reverse direction — concept injection — is equally interesting.

Representation engineering researchers have demonstrated that you can *add* a concept to a model by
adding its activation direction to the residual stream at inference time. For example, adding a
"confidence" direction makes the model sound more certain; adding a "formality" direction makes its
outputs more formal.

In Intelliton terms, concept injection is the operation of exciting a new Intelliton species that
was not activated by the input prompt. The Intelliton framework predicts that this will be most
stable when:

- the injected mode has low momentum (broad, sequence-level effect rather than token-local),
- the injected mode has low spin-like complexity (concentrated, easy to steer with a rank-1
  intervention),
- the injection is applied at the layer range where the mode has the lowest mass (highest
  propagation range).

These three conditions define a **tractability criterion** for representation engineering
interventions: not all concepts are equally steerable, and the Intelliton spectrum can predict which
ones are tractable before you attempt the intervention.

---

## Enterprise implications

The Abliteration/ARA episode revealed a commercially important fact: fine-tuning is not the only
way to customise an open-source model. Representation engineering with Intelliton-guided steering
maps could enable:

- **Domain-specific tone calibration** (formal, terse, verbose, empathetic) by identifying and
  amplifying or suppressing the relevant low-momentum, low-complexity style modes.
- **Compliance mode injection** (make a general model behave as if it were trained on a strict
  regulatory corpus) by injecting the compliance Intelliton species identified from a reference
  model.
- **Persona engineering** (the "Machiavellian" or "street punk" effect described in the Abliteration
  literature) by amplifying specific behavioural modes.

All of these operations require knowing *which modes to touch* and *at which layers*. The Intelliton
steering map is precisely that knowledge, expressed in a principled spectral language.

---

## The shortest summary

- Representation engineering steers behaviour by writing to the residual stream.
- The Intelliton framework characterises what is already in the residual stream.
- Together, they make it possible to identify *which modes to steer*, *how hard*, *at which layer*,
  and *at what cost to other modes*.
- The proposed Intelliton steering map would turn the species catalogue into a practical intervention
  guide for both safety-positive (alignment hardening) and safety-negative (jailbreaking) uses.

---

## Continue reading

- [Refusal as an Intelliton]({% post_url 2026-04-06-refusal-as-intelliton %})
- [Safety Alignment Through the Intelliton Lens]({% post_url 2026-04-08-safety-alignment-intelliton-landscape %})

</div>

<div data-lang="zh" markdown="1">

## 两个本该放在一起的概念

**表征工程**（Representation Engineering）是在不改变权重的前提下，直接读写模型内部激活，
从而引导模型行为的实践方法。大量研究表明，"快乐"、"权威"、"政治偏见"、"诚实"等概念可以
被编码为残差流中的线性方向，在推理时加上或减去这些方向的少量倍数，就能可靠地改变模型的
输出。

**Intelliton 框架**将残差流刻画为一个类准粒子模式的空间，提取反复出现的模式，并用谱属性标
注它们：动量、类自旋复杂度、质量和螺旋度。

这两套思想在描述同一个对象——残差流——只是抽象层次不同。表征工程说"这个方向引导这种行
为"；Intelliton 框架说"这个模式有这些谱属性"。把两者结合起来，两者都会变得更有用。

---

## 表征工程能做什么（以及它自己说不清什么）

与表征工程最相关的工具——激活叠加、对比激活分析，以及用于越狱 Gemma 4 的 ARA 方法——
有一个共同的局限：它们能确定*往哪个方向推*，但对于*那个方向在更广泛激活空间中的结构*却
几乎无话可说。

具体来说：

- **激活叠加**在每个 token 位置上，向选定层的残差流添加一个固定方向。对于简单概念，效果
  可靠，但当被引导方向与重要的任务求解模式重叠时，会降低模型性能。

- **对比激活分析**（Abliteration 的核心）计算两组对比激活的均值差，能高效找到拒绝方向，
  但无法告诉你涉及几个模式、这些模式的传播属性是什么，也无法说明它们与你想保留的任务
  求解模式有多少重叠。

- **ARA** 改进了简单的减法——它在低秩子空间而非单一方向上操作，用 SVD 分离拒绝子空间，
  但没有把分离出的各个分量与模型更广泛的模式景观联系起来。

Intelliton 框架恰好填补了这些空白。

---

## 引导地图的提案

本文提出的研究方向，是构建一张**Intelliton 引导地图**：一份对每个 Intelliton 物种标注其可
能行为角色、大致层范围、在残差流中的秩，以及与其他物种的重叠程度的目录。

### 构建地图：三个要素

**要素一 —— 任务探针**

使用 `src/datasets.py` 中的五类提示词（代词跟踪、事实回忆、逻辑推理、算术、句法一致性），
建立哪类 Intelliton 物种被哪类任务激活的对应关系。这部分已经在现有分析中有所涉及。

**要素二 —— 行为探针**

加入一类新探针，专门针对 RLHF 训练的行为：
- 拒绝（有害 vs. 无害提示词）；
- 讨好（奉承 vs. 中性提示词）；
- 政治中立性（争议性 vs. 中性框架）；
- 冗余度控制（指令要求简洁 vs. 指令要求详细的提示词）。

对每类行为探针集运行同样的 Intelliton 分析，记录哪些物种有响应。

**要素三 —— 跨探针重叠**

计算所有逐层拒绝向量与所有逐层任务激活向量之间的余弦相似度。在所有任务探针上重叠都低的
物种，是好的引导目标：增加或移除它们，不会渗透进任务性能。

---

## 与 ARA 的联系

ARA 技术构造一个秩为 $$k$$ 的惩罚矩阵 $$\Delta W$$，把拒绝子空间从模型权重矩阵中投影出
去。用 Intelliton 的语言说，$$\Delta W$$ 就是对少数几个 Intelliton 物种的定向抑制。

ARA 的核心主张是：对于具有强大推理能力的模型，更高秩的干预比秩-1 干预（简单向量减法）更
安全，因为拒绝行为跨越了多个纠缠的模式。如果只移除秩-1 分量，剩余分量会继续产生部分拒绝
或降低模型的推理能力。

这个主张可以用 Intelliton 框架直接检验：

1. 在有害提示词上计算指令微调模型的 Intelliton 谱；
2. 确定在有害提示词上最活跃、在无害提示词上最不活跃的模式；
3. 度量这些模式是否聚集在逐层 SVD 基的低维子空间里，还是分散在许多独立方向上。

如果它们是聚集的，ARA 的秩-$$k$$ 做法就是有依据的，而且可以在任何越狱尝试之前，通过
Intelliton 谱估算出聚集的秩 $$k$$。如果它们是分散的，简单减法预计会留下残余的拒绝能力，
或造成更广泛的附带损伤。

---

## 一个实际应用：零样本概念注入

反方向——概念注入——同样有趣。

表征工程研究人员已经证明，可以通过在推理时将某个概念的激活方向加入残差流，来*添加*这个
概念。例如，加入"自信"方向会让模型听起来更确定；加入"正式性"方向会让输出更正式。

用 Intelliton 的语言说，概念注入就是激发一个输入提示词本来没有激活的新 Intelliton 物种的
操作。Intelliton 框架预测，在以下情况下这种操作最稳定：

- 被注入的模式具有低动量（影响整段序列，而不是局部 token）；
- 被注入的模式具有低类自旋复杂度（内部集中，可用秩-1 干预轻松引导）；
- 注入发生在模式质量最低（传播范围最大）的层范围内。

这三个条件定义了表征工程干预的**可操作性判据**：不是所有概念都同样易于引导，而 Intelliton
谱可以在干预尝试之前就预测哪些概念是可操作的。

---

## 商业含义

Abliteration/ARA 事件揭示了一个对商业有重要意义的事实：微调不是定制开源模型的唯一途径。
基于 Intelliton 引导地图的表征工程或许能支持：

- **特定领域语气校准**（正式、简洁、详尽、移情），通过识别并放大或抑制相关的低动量、低
  复杂度风格模式；
- **合规模式注入**（让通用模型表现得像在严格监管语料上训练过），通过从参考模型中识别出
  合规 Intelliton 物种并注入；
- **人格工程**（Abliteration 文献中描述的"马基雅维利型"或"街头混混型"效果），通过放大特定
  行为模式。

所有这些操作都需要知道*该动哪些模式*以及*在哪些层操作*。Intelliton 引导地图，正是以原则
性谱语言表达出来的那份知识。

---

## 最短总结

- 表征工程通过写入残差流来引导行为；
- Intelliton 框架刻画残差流中已经存在的内容；
- 两者结合，就能确定*该引导哪些模式*、*力度多大*、*在哪一层*，以及*对其他模式的代价*；
- 提议的 Intelliton 引导地图，将把物种目录变成一份可操作的干预指南，对安全正向（加固对齐）
  和安全负向（越狱）两类用途都适用。

---

## 继续阅读

- [拒绝即 Intelliton]({% post_url 2026-04-06-refusal-as-intelliton %})
- [用 Intelliton 视角看安全对齐]({% post_url 2026-04-08-safety-alignment-intelliton-landscape %})

</div>
