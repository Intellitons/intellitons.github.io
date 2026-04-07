---
layout: post
title: "Safety Alignment Through the Intelliton Lens: Toward Structural Guarantees"
title_zh: "用 Intelliton 视角看安全对齐：迈向结构性保证"
date: 2026-04-08
categories: [safety, alignment, research-directions]
excerpt: >
  RLHF-based alignment has been shown to be a thin spectral overlay that can be removed in minutes
  on any open-source model. This article argues that the Intelliton framework offers a route toward
  something more robust: structural alignment — where safety-relevant modes are architecturally
  entangled with capability modes, making removal costly rather than free.
excerpt_zh: >
  基于 RLHF 的对齐已经被证明只是一层薄薄的谱覆盖，可以在几分钟内从任何开源模型上被移除。
  本文认为，Intelliton 框架提供了一条通向更稳健方向的道路：结构性对齐——在这种对齐中，安全
  相关模式在架构层面与能力模式相互纠缠，从而使移除变得代价高昂而非轻而易举。
---

<div data-lang="en" markdown="1">

## The uncomfortable lesson from Gemma 4

The ARA jailbreak of Gemma 4 in April 2026 demonstrated something that the AI safety community
had long feared but struggled to quantify: **RLHF-imposed alignment is not a deep architectural
property of the model — it is a separable spectral overlay.**

The implication is stark. Any open-source model, no matter how carefully aligned during training,
can have its alignment stripped by someone with:
- access to the model weights,
- a few hundred forward passes to collect contrast activations,
- a laptop and a few minutes of linear algebra.

This is not a failure of any particular alignment method. It is a structural property of how
current RLHF and DPO work: they shift the model's behavioural outputs by adjusting the magnitudes
and directions of a small set of residual-stream modes, but they do not fundamentally restructure
the mode landscape inherited from pre-training.

---

## What "thin alignment" looks like in Intelliton terms

The alignment-vs-base comparison in
[Scaling and Alignment Through the Intelliton Lens]({% post_url 2026-04-03-scaling-alignment-intellitons %})
shows that instruction tuning creates measurable but limited changes to the Intelliton spectrum:

- a shift in dominant momentum (the alignment process changes the sequence-scale structure of the
  main backbone mode),
- a modest reduction in the number of species (some modes are suppressed or merged),
- a shift in fixed-point structure (all points become crossovers, suggesting a more uniform
  propagation regime).

What does *not* change is the fundamental mode landscape. The same species types appear in both the
base and instruct models. The instruct model has had some modes adjusted and one or two new ones
added, but the bulk of the residual-stream dynamics are inherited directly from pre-training.

In Abliteration terms, this means the refusal modes sit *on top of* the task-solving modes rather
than being *woven into* them. Removing the refusal modes does not substantially disturb the
task-solving modes, which is why jailbroken models retain their capabilities.

---

## The structural alignment hypothesis

The Intelliton framework suggests a more robust alignment paradigm:

> **Structural alignment**: instead of adding refusal modes on top of the existing mode landscape
> (current RLHF), train the model such that safety-relevant mode properties are *architecturally
> entangled* with capability-relevant modes across many layers and many tasks.

Under structural alignment, removing a safety mode would necessarily degrade a capability mode,
because the two would share subspace components across multiple layers. The cost of ablation rises
from near-zero to a genuine capability penalty.

This is an analogy to the cancer treatment metaphor in the ARA literature: instead of making
cancer cells identifiable for targeted removal (current RLHF), make them biologically inseparable
from healthy tissue in a way that deters removal.

---

## Three concrete research directions

### Direction 1 — Measure current alignment depth

The first step is to quantify how separable alignment modes actually are, using the Intelliton
framework as the measurement instrument.

**Protocol:**
1. For a matched pair of base and instruct models (e.g., `Qwen3-8B-Base` and `Qwen3-8B`), compute
   the per-layer Intelliton species catalogue for both.
2. For each species in the instruct model, compute its cosine similarity with the closest species
   in the base model at the same layer.
3. Define an **alignment depth score** as the fraction of alignment-specific modes (modes present
   in the instruct model but not in the base model, or modes with significantly shifted spectral
   properties) that have low cosine overlap with *all* task-solving modes.

A high alignment depth score means alignment modes are deeply entangled with task modes
(structurally hard to remove). A low score means they are orthogonal (structurally easy to remove).

The hypothesis is that current RLHF produces a low alignment depth score, and that this is
measurable with the existing Intelliton toolkit before any jailbreak attempt.

### Direction 2 — Design training objectives that increase alignment depth

If alignment depth is measurable, it becomes a trainable objective.

The proposed training signal would add a **mode entanglement regularisation term** to the RLHF or
DPO loss. The term penalises configurations where safety-relevant mode directions are orthogonal to
capability-relevant mode directions at the same layer:

$$\mathcal{L}_{\text{entanglement}} = -\sum_{\ell} \sum_{s \in \text{safety}} \sum_{c \in \text{capability}} \left| \langle \hat{v}_{s,\ell}, \hat{v}_{c,\ell} \rangle \right|$$

Minimising this term (as a penalty) during alignment training would push the model toward
configurations where safety modes share subspace components with capability modes — increasing the
cost of surgical removal.

This is speculative, but it is testable at small scale on the models already analysed by the
Intelliton project.

### Direction 3 — Use Intelliton audits as a pre-deployment safety check

Even before structural alignment is achievable, the Intelliton framework can be used as a
**pre-deployment safety audit** for open-source models.

The audit would:
1. Run the Intelliton analysis on the released model with contrast prompt sets.
2. Compute the alignment depth score.
3. Report the estimated minimum cost of abliteration (how many modes need to be removed, what the
   expected capability penalty is).

This would give the open-source community a standardised, interpretable metric for alignment
robustness — something that is currently entirely absent from model release documentation.

---

## The deeper issue: "teach the model to not say" versus "teach the model to not know"

The Abliteration literature makes a pointed observation that maps directly onto the Intelliton
framework:

> "Teaching the model not to say" (current RLHF) can be defeated. "Teaching the model not to know"
> (removing the capability from the pre-training stage) cannot be defeated by post-hoc ablation.

In Intelliton terms:

- **Behavioural alignment** (current RLHF) adds a small number of low-complexity, separable refusal
  modes that sit orthogonally to the capability modes. These can be removed with targeted ablation.
- **Capability-level safety** would require that certain capability modes — the ones that underlie
  dangerous knowledge — are never formed during pre-training, or are formed in such a way that they
  are deeply entangled with unrelated benign modes.

The Intelliton framework cannot, by itself, implement capability-level safety. But it can *measure*
the difference: a model with capability-level safety for a particular dangerous capability would
show, under Intelliton analysis, that the dangerous-knowledge modes are spectrally entangled with
benign modes in a way that makes targeted removal impossible without broad capability degradation.

This becomes a falsifiable, quantitative prediction that can be tested on released models.

---

## Why Bengio's warning deserves a technical interpretation

Yoshua Bengio, one of the three godfathers of deep learning, has consistently argued that open-sourcing powerful models is dangerous, because once the weights are released, the alignment can be
removed by anyone with modest technical resources.

The Intelliton framework gives that warning a technical, measurable form:

> **A model's alignment robustness is bounded above by its alignment depth score. Current models,
> based on the spectral evidence already available from base/instruct comparisons, have low
> alignment depth scores.**

This is not a political statement. It is a quantitative prediction that can be tested, and that, if
it holds, tells us that the current open-source release paradigm for aligned models carries
measurable safety risks that can be expressed in the language of residual-stream spectral analysis.

---

## The research agenda in summary

| Step | What to measure | What it tells us |
|------|----------------|-----------------|
| Alignment depth audit | Cosine overlap between safety modes and task modes, per layer | How separable current alignment is |
| Alignment depth score across model families | Score vs. model size, RLHF method, training data | What factors increase structural alignment |
| Entanglement regularisation experiment | Alignment depth score before and after training with mode-entanglement loss | Whether structural alignment is trainable |
| Pre-deployment audit protocol | Standardised depth score at release time | A public, interpretable alignment robustness metric |

Each of these steps is feasible using the infrastructure already developed in the Intelliton
project. The first step requires only a new set of contrast prompts and a small extension to the
existing analysis pipeline.

---

## The shortest summary

- Current RLHF alignment is spectrally thin: alignment modes are separable from capability modes,
  and this separability is what makes Abliteration/ARA work.
- The Intelliton framework can measure this separability as a quantitative **alignment depth score**.
- A research direction based on this measurement would pursue **structural alignment** — training
  objectives that increase mode entanglement and make ablation genuinely costly.
- Even before structural alignment is achieved, the Intelliton audit provides a standardised
  pre-deployment robustness metric that is currently entirely absent.

---

## Continue reading

- [Refusal as an Intelliton]({% post_url 2026-04-06-refusal-as-intelliton %})
- [Representation Engineering and Intelliton Steering]({% post_url 2026-04-07-representation-engineering-intelliton-steering %})

</div>

<div data-lang="zh" markdown="1">

## Gemma 4 带来的令人不安的教训

2026 年 4 月 Gemma 4 遭受 ARA 越狱，向 AI 安全社区证明了一件他们早已担忧却难以量化的事：
**RLHF 引入的对齐，不是模型的深层架构属性，而是一个可分离的谱覆盖层。**

这意味着，不管在训练中做了多少精心的对齐工作，任何开源模型都可以被拥有以下资源的人剥除
对齐：
- 模型权重的访问权限；
- 几百次前向传播，用于收集对比激活；
- 一台笔记本电脑和几分钟的线性代数运算。

这不是某种特定对齐方法的失败，而是当前 RLHF 和 DPO 工作方式的结构性属性：它们通过调整
少数几个残差流模式的量级和方向来改变模型的行为输出，但并没有从根本上重构继承自预训练的
模式景观。

---

## "薄对齐"在 Intelliton 语言里长什么样

[用 Intelliton 视角看规模扩展与对齐]({% post_url 2026-04-03-scaling-alignment-intellitons %})
中的对比表明，指令微调确实对 Intelliton 谱产生了可测量但有限的变化：

- 主导动量发生偏移（对齐过程改变了主干模式的序列尺度结构）；
- 物种数量小幅减少（一些模式被抑制或合并）；
- 不动点结构发生变化（所有不动点变成 crossover，意味着更均匀的传播机制）。

没有改变的是基本的模式景观。base 模型和 instruct 模型中出现的物种类型相同。instruct 模型
对一些模式做了调整，添加了一两个新模式，但残差流动力学的主体是直接从预训练继承而来的。

用 Abliteration 的语言说，这意味着拒绝模式是*叠加在*任务求解模式之上的，而不是*编织进*
任务求解模式里的。移除拒绝模式不会实质性地扰乱任务求解模式，这就是越狱模型仍然保有能力
的原因。

---

## 结构性对齐假说

Intelliton 框架提示了一种更稳健的对齐范式：

> **结构性对齐**：不是把拒绝模式叠加在已有模式景观之上（当前 RLHF），而是训练模型，使安全
> 相关的模式属性在架构层面与能力相关的模式*相互纠缠*，遍布多层、多任务。

在结构性对齐下，移除一个安全模式必然会损害一个能力模式，因为两者在多个层上共享子空间分
量。消融的代价从接近零，上升为真实的能力损失。

这与 ARA 文献中的癌细胞治疗类比相似：不是把癌细胞标记出来以便精准切除（当前 RLHF），而
是让它们在生物学上与健康组织不可分离，从而从根本上阻止切除。

---

## 三个具体研究方向

### 方向一 —— 测量当前对齐深度

第一步是量化当前对齐模式的实际可分离程度，以 Intelliton 框架作为测量工具。

**方案：**
1. 对一对匹配的 base/instruct 模型（例如 `Qwen3-8B-Base` 和 `Qwen3-8B`），分别计算逐层的
   Intelliton 物种目录；
2. 对 instruct 模型中每个物种，计算其与 base 模型同一层最近邻物种的余弦相似度；
3. 将**对齐深度分数**定义为：对齐专属模式（出现在 instruct 但不在 base 中，或谱属性发生显
   著偏移的模式）中，与所有任务求解模式的余弦重叠都低的那部分比例。

对齐深度分数高，说明对齐模式与任务模式深度纠缠（结构上难以移除）；分数低，说明它们是正
交的（结构上易于移除）。

假设是：当前 RLHF 产生的对齐深度分数偏低，而且这可以用现有的 Intelliton 工具集在任何越狱
尝试之前就测量出来。

### 方向二 —— 设计能提升对齐深度的训练目标

如果对齐深度可以被测量，它就可以成为一个可训练的目标。

提议的训练信号，是在 RLHF 或 DPO 损失中加入一个**模式纠缠正则化项**。该项惩罚安全相关模
式方向与同层能力相关模式方向正交的配置：

$$\mathcal{L}_{\text{entanglement}} = -\sum_{\ell} \sum_{s \in \text{safety}} \sum_{c \in \text{capability}} \left| \langle \hat{v}_{s,\ell}, \hat{v}_{c,\ell} \rangle \right|$$

在对齐训练中最小化这一惩罚项，会推动模型朝向安全模式与能力模式共享子空间分量的配置——
从而提高外科手术式移除的代价。

这是一个推测性方向，但可以在 Intelliton 项目已经分析过的小规模模型上加以检验。

### 方向三 —— 把 Intelliton 审计用作部署前安全检查

即使在结构性对齐尚未实现之前，Intelliton 框架也可以用作开源模型的**部署前安全审计**工具。

审计流程包括：
1. 用对比提示词集对发布模型运行 Intelliton 分析；
2. 计算对齐深度分数；
3. 报告消融的估计最低代价（需要移除多少模式，预期能力损失是多少）。

这将为开源社区提供一个标准化、可解释的对齐鲁棒性指标——而这正是目前模型发布文档中完全
缺失的东西。

---

## 更深层的问题："教模型不说"与"教模型真不懂"

Abliteration 文献中有一个直接映射到 Intelliton 框架的深刻观察：

> "教模型不说"（当前 RLHF）可以被攻破。"教模型真不懂"（在预训练阶段就移除该能力）无法
> 被事后消融攻破。

用 Intelliton 的语言说：

- **行为对齐**（当前 RLHF）添加了少数低复杂度、可分离的拒绝模式，它们与能力模式正交。
  这些模式可以用定向消融移除。
- **能力层面的安全**，则需要某些能力模式——那些支撑危险知识的模式——从未在预训练中形
  成，或者以与其他良性模式深度纠缠的方式形成，使得定向移除不可能在不引发大规模能力退化
  的情况下完成。

Intelliton 框架本身无法实现能力层面的安全，但它能*测量*这种差异：一个针对某种特定危险能
力实现了能力层面安全的模型，在 Intelliton 分析下会表现出，危险知识模式与良性模式在谱上的
纠缠程度，使得任何定向移除都不可能在不引发广泛能力退化的情况下完成。

这成为了一个可检验的、定量的预测，可以在已发布模型上加以验证。

---

## 为什么本吉奥的警告值得一个技术性解读

深度学习三巨头之一的 Yoshua Bengio，一直坚持认为开源强大模型是危险的，因为一旦权重被公开，
任何拥有适度技术资源的人都可以移除对齐。

Intelliton 框架为这一警告赋予了技术性、可量化的形式：

> **模型的对齐鲁棒性，其上界就是它的对齐深度分数。根据 base/instruct 对比中已有的谱证据，
> 当前模型的对齐深度分数偏低。**

这不是政治表态，而是一个可以检验的定量预测。如果它成立，就告诉我们：当前已对齐模型的开
源发布范式，携带着可测量的安全风险，而这些风险可以用残差流谱分析的语言来表达。

---

## 研究议程总结

| 步骤 | 测量什么 | 告诉我们什么 |
|------|----------|------------|
| 对齐深度审计 | 安全模式与任务模式的逐层余弦重叠 | 当前对齐的可分离程度 |
| 跨模型家族对齐深度分数 | 分数 vs. 模型大小、RLHF 方法、训练数据 | 哪些因素提升结构性对齐 |
| 纠缠正则化实验 | 训练前后的对齐深度分数 | 结构性对齐是否可训练 |
| 部署前审计协议 | 发布时的标准化深度分数 | 公开的、可解释的对齐鲁棒性指标 |

上述每个步骤，用 Intelliton 项目已有的基础设施都是可行的。第一步只需要一组新的对比提示词，
以及对现有分析流程的少量扩展。

---

## 最短总结

- 当前 RLHF 对齐在谱层面是薄的：对齐模式与能力模式可分离，而这种可分离性正是
  Abliteration/ARA 得以奏效的原因；
- Intelliton 框架能将这种可分离性量化为**对齐深度分数**；
- 基于这一测量的研究方向，将追求**结构性对齐**——提升模式纠缠程度、使消融变得真正代价
  高昂的训练目标；
- 即使在结构性对齐尚未实现之前，Intelliton 审计也提供了一个标准化的部署前鲁棒性指标，而
  这正是目前完全缺失的。

---

## 继续阅读

- [拒绝即 Intelliton]({% post_url 2026-04-06-refusal-as-intelliton %})
- [表征工程与 Intelliton 引导]({% post_url 2026-04-07-representation-engineering-intelliton-steering %})

</div>
