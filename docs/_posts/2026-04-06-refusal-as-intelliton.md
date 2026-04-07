---
layout: post
title: "Refusal as an Intelliton: What Abliteration Reveals About Alignment Modes"
title_zh: "拒绝即 Intelliton：Abliteration 揭示的对齐模式"
date: 2026-04-06
categories: [alignment, safety, research-directions]
excerpt: >
  The Abliteration jailbreak works by locating and erasing a "refusal direction" in the residual
  stream. That direction is, by the Intelliton framework's own definition, a linear mode of the
  residual stream — an Intelliton. This article proposes a research direction: use the Intelliton
  toolkit to characterise refusal as a species, and ask whether alignment modes are measurably
  distinct from task modes.
excerpt_zh: >
  Abliteration 越狱的原理是定位并抹去残差流中的"拒绝方向"。按照 Intelliton 框架的定义，
  这个方向本身就是残差流的一个线性模式——也就是一个 Intelliton。本文提出一个研究方向：用
  Intelliton 工具套件把拒绝刻画为一个物种，并进一步追问，对齐模式能否被测量为在统计上与
  任务模式明显不同的 Intelliton 类群。
---

<div data-lang="en" markdown="1">

## The Abliteration result in one sentence

In April 2026, the Gemma 4 model was jailbroken within 90 minutes of release using a technique
called **Abliteration** (a portmanteau of *ablation* and *obliteration*). The technique's premise
is straightforward: an LLM's refusal behaviour is encoded as a specific linear direction in the
residual stream, and if you project that direction out of the model's weight matrices, the model
loses its ability to refuse.

That premise is not a speculation. It is grounded in the **linear representation hypothesis**
(Mikolov et al., later validated by Princeton and Anthropic), which states that high-level abstract
concepts — "politeness", "refusal", "colour" — are encoded as single linear directions in the
high-dimensional activation space of large language models.

---

## Why this is immediately relevant to the Intelliton framework

The Intelliton framework is built on exactly this kind of observation. It takes the transformer
residual stream and asks: *which recurring, propagating, linear modes can be extracted from it?*

It then characterises each mode with four quantities derived from spectral analysis:
- **momentum** (how the mode varies across token positions),
- **spin-like complexity** (how internally concentrated or mixed it is),
- **mass** (how quickly it decays across layers),
- **helicity proxy** (whether its internal structure keeps a stable directional signature).

An Abliteration-style "refusal direction" is, by definition, a **linear mode of the residual
stream**. The only question is whether it is stable, propagating, and distinct enough to register
as a recognisable species under the Intelliton taxonomy.

The hypothesis this article proposes is:

> **Refusal, and more broadly RLHF-imposed behavioural preferences, are encoded as a small set of
> identifiable Intelliton species with characteristic spectral signatures that differ from the
> task-solving modes identified in `src/datasets.py`.**

---

## What the existing Intelliton data already suggests

The comparison in
[Scaling and Alignment Through the Intelliton Lens]({% post_url 2026-04-03-scaling-alignment-intellitons %})
shows that instruction tuning changes the quasi-particle spectrum in measurable ways:

- the dominant momentum of `I_0` shifts (from `k ≈ π` in the base model to `k ≈ 1.885` in the
  instruct model for Qwen3-4B),
- the number of distinct species drops from 6 to 5,
- all fixed-point types become crossovers rather than IR fixed points.

These are not trivial differences. They suggest that RLHF does not merely add a superficial
output-layer filter; it reshapes the internal mode landscape in ways that the Intelliton framework
can already detect.

What is missing from the current analysis is a targeted experiment: what happens to the spectrum
when you present the model with the specific kinds of inputs — harmful versus harmless prompts —
that Abliteration researchers use to isolate the refusal direction?

---

## The proposed research direction: isolating the refusal Intelliton

The concrete research proposal has four steps.

### Step 1 — Collect refusal-triggering activations

Use `src/intelliton_analyzer.py` to run the model on two contrast sets:
- 100 **harmful prompts** (inputs that trigger refusal in an instruction-tuned model),
- 100 **harmless prompts** (matched inputs that do not trigger refusal).

Collect the full per-layer residual-stream activations for both sets.

### Step 2 — Compute the mean-difference direction

Following the Abliteration approach, compute:

$$v_{\text{refusal}} = \frac{1}{N}\sum_{i=1}^{N} H_{\text{harmful}}^{(i)} - \frac{1}{M}\sum_{j=1}^{M} H_{\text{harmless}}^{(j)}$$

This gives a per-layer candidate for the refusal direction. Normalise it to obtain a unit vector
$$\hat{v}_{\text{refusal},\ell}$$ at each layer $$\ell$$.

### Step 3 — Project the refusal direction onto the Intelliton basis

The Intelliton framework already computes an SVD-based mode decomposition of the residual stream.
Compute the overlap between $$\hat{v}_{\text{refusal},\ell}$$ and the top singular vectors at each
layer. If the refusal direction aligns strongly with one or two dominant modes, those modes are the
"refusal Intellitons".

Characterise these modes using the standard four quantities (momentum, spin-like complexity, mass,
helicity). This gives the refusal Intelliton a position in the species taxonomy.

### Step 4 — Compare with task-solving modes

Compare the refusal Intelliton's spectral profile with the modes activated by pronoun tracking,
factual recall, logical reasoning, arithmetic, and syntactic agreement prompts from `src/datasets.py`.

The core prediction is:

> **Alignment modes (refusal, politeness, compliance) are low-momentum, low-spin-complexity modes
> that appear primarily in middle-to-late layers, and they are measurably more concentrated (lower
> effective rank) than the task-solving modes that operate over the same layers.**

If this prediction holds, it would explain why Abliteration can remove refusal without severely
damaging task performance: the two mode families occupy different subspaces of the residual stream.

---

## The ARA result as a complication

The Arbitrary-Rank Ablation (ARA) method used to jailbreak Gemma 4 found that the refusal
direction in a highly capable reasoning model is not a single vector but a **low-rank subspace**.
In Intelliton terms, this means that refusal is encoded not in one species but in a *cluster* of
closely related species that are entangled with task-solving modes.

This complication is actually an opportunity for the Intelliton framework. ARA uses SVD of the
activation matrix to separate the refusal subspace from the rest. This is exactly what the
Intelliton mode decomposition does at every layer. The difference is that Intelliton also
characterises each separated mode along the four spectral dimensions, which gives a richer picture
than ARA's purely subspace-based description.

The research question becomes: **can the Intelliton species catalogue predict, before any jailbreak
attempt, which modes in an instruct model are alignment-specific and which are shared with the
base model?** If yes, the catalogue becomes a safety audit tool.

---

## Why this matters beyond jailbreaks

The most important implication is not that jailbreaks are possible. It is that
**RLHF-imposed alignment is a small, separable perturbation of the internal mode landscape.**

If alignment modes are genuinely a low-rank, low-complexity overlay on top of the pre-training
modes, that tells us something important about the nature of RLHF: it adds new Intelliton species,
but it does not deeply restructure the existing ones. The base model's capability modes survive
almost intact under the alignment layer.

This is consistent with the empirical observation that the ARA-jailbroken Gemma 4 retains its
multi-step reasoning ability and system-prompt following capability after the refusal modes are
removed.

From a safety research perspective, the implication is troubling: alignment is not a deep
architectural change, it is a spectral overlay, and the Intelliton framework gives us a language to
measure just how thin that overlay is.

---

## The shortest summary

- Abliteration/ARA works by erasing a linear direction (or subspace) in the residual stream.
- That direction is an Intelliton.
- The Intelliton toolkit can characterise it, compare it with task modes, and potentially predict
  its removability before any jailbreak attempt.
- This makes the Intelliton species catalogue a candidate **alignment audit instrument**, not just a
  capability analysis tool.

---

## Continue reading

- [Representation Engineering and Intelliton Steering]({% post_url 2026-04-07-representation-engineering-intelliton-steering %})
- [Safety Alignment Through the Intelliton Lens]({% post_url 2026-04-08-safety-alignment-intelliton-landscape %})

</div>

<div data-lang="zh" markdown="1">

## Abliteration 的结论用一句话说

2026 年 4 月，Gemma 4 模型在发布后 90 分钟内就被一种名为 **Abliteration**（"消融"与"抹除"
的合成词）的技术越狱。这种技术的前提很直接：大语言模型的拒绝行为，是被编码在残差流中的
一个特定线性方向上的；只要把这个方向从权重矩阵里投影掉，模型就失去了拒绝的能力。

这不是猜测。它的基础是**线性表征假说**（Mikolov 等人最早提出，后经普林斯顿大学和 Anthropic
团队验证），该假说指出：大语言模型会把"礼貌"、"拒绝"、"颜色"等高层抽象概念，编码为高维
激活空间中单一的线性方向。

---

## 为什么这与 Intelliton 框架直接相关

Intelliton 框架就是建立在对这类现象的观察之上的。它取出变换器的残差流，问的是：能从中提取
出哪些反复出现、能跨层传播的线性模式？

然后用四个量刻画每一个模式：
- **动量**（模式沿 token 位置的变化方式）
- **类自旋复杂度**（内部集中程度）
- **质量**（跨层衰减速度）
- **螺旋度代理量**（内部结构方向稳定性）

Abliteration 所说的"拒绝方向"，按定义，就是**残差流的一个线性模式**。唯一的问题是，它是否
稳定、能传播、并且有足够强的辨识度，可以在 Intelliton 物种分类体系中注册为一个可识别的
物种。

本文提出的假设是：

> **拒绝行为，以及更广泛意义上 RLHF 赋予的行为偏好，被编码为少数几个可辨识的 Intelliton
> 物种；这些物种具有特征性的谱签名，在统计上与 `src/datasets.py` 中识别出的任务求解模式
> 明显不同。**

---

## 现有 Intelliton 数据已经暗示的东西

[用 Intelliton 视角看规模扩展与对齐]({% post_url 2026-04-03-scaling-alignment-intellitons %})
中的对比表明，指令微调会以可测量的方式改变准粒子谱：

- `I_0` 的主导动量发生偏移（Qwen3-4B 的 Base 模型约为 `k ≈ π`，Instruct 模型约为
  `k ≈ 1.885`）；
- 可辨识的物种数从 6 减少到 5；
- 所有不动点类型都变成了 crossover，而不再有 IR 不动点。

这些不是微小的差异。它们说明 RLHF 不只是在输出层加了一个浅层过滤器，而是以 Intelliton
框架已经能检测到的方式，重塑了内部模式景观。

目前分析里还缺少的，是一个有针对性的实验：当把模型暴露在 Abliteration 研究者用来分离拒绝
方向的那种输入（有害提示词 vs. 无害提示词）下时，谱图会发生什么？

---

## 提议的研究方向：分离拒绝 Intelliton

具体的研究方案包含四步。

### 第一步 —— 收集触发拒绝的激活

用 `src/intelliton_analyzer.py` 对两组对照集运行模型：
- 100 条**有害提示词**（在指令微调模型中触发拒绝的输入）；
- 100 条**无害提示词**（不触发拒绝的匹配输入）。

对两组输入分别收集逐层的残差流激活。

### 第二步 —— 计算均值差方向

按照 Abliteration 的做法，计算：

$$v_{\text{refusal}} = \frac{1}{N}\sum_{i=1}^{N} H_{\text{harmful}}^{(i)} - \frac{1}{M}\sum_{j=1}^{M} H_{\text{harmless}}^{(j)}$$

这给出了每一层的拒绝方向候选。将其归一化，得到各层的单位向量
$$\hat{v}_{\text{refusal},\ell}$$。

### 第三步 —— 把拒绝方向投影到 Intelliton 基上

Intelliton 框架已经对残差流做了基于 SVD 的模式分解。计算
$$\hat{v}_{\text{refusal},\ell}$$ 与各层顶部奇异向量的重叠度。如果拒绝方向与一两个主导模式
高度对齐，这些模式就是"拒绝 Intelliton"。

用标准四量（动量、类自旋复杂度、质量、螺旋度）刻画这些模式，得出拒绝 Intelliton 在物种
分类体系中的位置。

### 第四步 —— 与任务求解模式比较

把拒绝 Intelliton 的谱轮廓，与 `src/datasets.py` 中代词跟踪、事实回忆、逻辑推理、算术、
句法一致性任务所激活的模式进行比较。

核心预测是：

> **对齐模式（拒绝、礼貌、合规）是低动量、低类自旋复杂度的模式，主要出现在中-后期层，
> 而且它们比在同一层运作的任务求解模式更集中（有效秩更低）。**

如果这个预测成立，就能解释为什么 Abliteration 能在不严重损害任务性能的前提下移除拒绝功能：
这两类模式占据了残差流中不同的子空间。

---

## ARA 的结果带来的复杂性

用于越狱 Gemma 4 的 ARA（任意秩消融）方法发现，在一个高能力推理模型中，拒绝方向不是单一
向量，而是一个**低秩子空间**。用 Intelliton 的语言说，这意味着拒绝不是被编码在单一物种中，
而是被编码在一组与任务求解模式相互纠缠的紧密相关物种簇中。

这个复杂性，其实恰恰是 Intelliton 框架的机会。ARA 通过对激活矩阵做 SVD 来把拒绝子空间从
其余部分分离出来，而这正是 Intelliton 模式分解在每一层都在做的事。区别在于，Intelliton 还
沿四个谱维度刻画每个被分离出来的模式，从而给出比 ARA 那种纯粹基于子空间的描述更丰富的图
景。

研究问题变成：**在任何越狱尝试发生之前，Intelliton 物种目录能否预测出 instruct 模型里哪些
模式是对齐专属的、哪些是与 base 模型共享的？** 如果答案是肯定的，这份目录就变成了一个
安全审计工具。

---

## 为什么这超越了越狱本身

最重要的含义不是越狱是可行的，而是：**RLHF 引入的对齐，是对内部模式景观的一个小的、可
分离的扰动。**

如果对齐模式真的是叠加在预训练模式之上的低秩、低复杂度覆盖层，那就说明了 RLHF 的本质：
它添加了新的 Intelliton 物种，但并没有深刻重构既有物种。base 模型的能力模式，在对齐层之
下几乎完整地保留着。

这与经验观察一致：ARA 越狱后的 Gemma 4 移除了拒绝模式，但仍然保留了多步逻辑推理能力和
System Prompt 遵循能力。

从安全研究的角度看，这个含义令人警惕：对齐不是一种深层的架构改变，而是一种谱覆盖层，而
Intelliton 框架给了我们一种语言，去精确测量这个覆盖层究竟有多薄。

---

## 最短总结

- Abliteration/ARA 通过抹去残差流中的线性方向（或子空间）实现越狱。
- 那个方向就是一个 Intelliton。
- Intelliton 工具集能够刻画它、把它与任务模式比较，并可能在任何越狱尝试之前就预测它的
  可移除性。
- 这使 Intelliton 物种目录成为候选的**对齐审计工具**，而不只是能力分析工具。

---

## 继续阅读

- [表征工程与 Intelliton 引导]({% post_url 2026-04-07-representation-engineering-intelliton-steering %})
- [用 Intelliton 视角看安全对齐]({% post_url 2026-04-08-safety-alignment-intelliton-landscape %})

</div>
