---
layout: post
title: "Inside Qwen3-4B-Base: Mapping a Language Model's Internal Particle Spectrum"
title_zh: "走进 Qwen3-4B-Base：描绘语言模型的内部粒子谱"
date: 2026-04-02
categories: [case-study, qwen]
excerpt: >
  The clearest single demonstration of the Intelliton framework comes from Qwen3-4B-Base. This
  article walks through its complete Intelliton catalogue — 6 species with masses, momenta,
  fixed-point layers, and distinct task affinities — and explains what each finding means in plain
  language.
excerpt_zh: >
  Intelliton 框架最清晰的一次单模型展示来自 Qwen3-4B-Base。本文按图索骥地走完整个
  Intelliton 目录：6 个物种、各自的质量、动量、固定点层与任务偏好，并用尽量直白的语言
  解释这些结果意味着什么。
---

<div data-lang="en" markdown="1">

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
2. **Treats the activation array as a discrete field** on a token-layer lattice.
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

The amplitude of `I_0` is about 30x larger than any other species. That makes it the dominant
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

The coexistence of `I_0` at high momentum (k ≈ π) and `I_1`-`I_5` at low momentum (k ≈ 0) is one
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
mid-to-late layers (around layer 13-16), consistent with the RG fixed-point analysis.

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

</div>

<div data-lang="zh" markdown="1">

## 最清晰的一扇窗口：看见 Intelliton 结构

在 Intelliton 项目分析的五个模型里，`Qwen3-4B-Base` 给出了最干净、也最容易解释的结果。
它是 Qwen3 家族的一个 40 亿参数基础模型，没有经过指令微调，也没有对齐处理，因此它的
内部结构更接近无监督预训练自然形成的组织方式。

这篇文章会完整走一遍这个模型最重要的发现：粒子目录、动量结构、重整化群流，以及生成
过程中的动力学。看完之后，你会对“一个 Intelliton 目录在实践中长什么样”有一个具体印象。

---

## 这张目录是怎么做出来的

分析从模型的 **残差流** 开始，也就是贯穿每一层的隐藏激活向量。对一组覆盖算术、事实回
忆、逻辑推理、代词跟踪和句法一致性的基准提示词，代码会执行以下步骤：

1. **捕获每一层、每一个 token 的残差激活**；
2. **把激活数组视作 token-layer lattice 上的离散场**；
3. **对 token 位置做傅里叶分析**，提取主导空间模态，也就是动量；
4. **逐层做奇异值分解（SVD）**，找出主导的集体模式；
5. **沿层方向拟合传播子**，为每个模式提取有效质量；
6. **运行 EFT / RG 分析**，识别固定点层，并把模式分成 UV、IR 或 crossover；
7. **跨提示词合并相似模式**，形成最终的物种目录。

最终得到的是一张紧凑的表，也就是 Intelliton 目录。相对于模型数十亿原始参数，它用少得多
的数字概括了内部的主导结构。

![Qwen3-4B-Base particle table]({{ "/assets/images/Qwen3-4B-Base/particle_table.png" | relative_url }})
*Qwen3-4B-Base 的 Intelliton 目录。共 6 个物种，每个都有质量、动量、类自旋分数、振幅、
固定点层和任务偏好。*

---

## 不是几百个物种，而是只有六个

最令人惊讶的结果之一，是目录的紧凑程度。在这条分析管线上，`Qwen3-4B-Base` 最终只分解
出 **6 个主导物种**：`I_0` 到 `I_5`。

这并不意味着模型简单。它仍然有 40 亿参数和丰富的激活空间。但这意味着，大多数系统性、
可重复出现的集体行为，实际上集中在少数几个主导模式上。就像复杂等离子体的许多现象，
往往主要由少数主导等离激元模态描述一样。

### 最领先的物种：`I_0`

优势最大的物种是 `I_0`：

| 属性 | 数值 |
|---|---|
| 振幅 | **6167.1** |
| 类自旋分数 | **1.84** |
| 极点质量 | **0.1275** |
| 格点质量 | **0.0083** |
| 动量 | **π ≈ 3.14** |
| 固定点层 | **16** |
| 固定点类型 | **crossover** |
| 活跃任务 | arithmetic, factual recall, logical reasoning |

`I_0` 的振幅大约是其他任何物种的 30 倍，因此它可以被看作模型的主导 **骨干模态**。

这些属性分别说明了什么？

**振幅（6167.1）** 表示这个模式被激活得非常强。模型处理提示词时，`I_0` 是最显著的集体
激发。

**类自旋分数（1.84）** 衡量内部复杂度。接近 1 的值意味着模式较为简单、相干；更高则说
明内部结构更丰富。`I_0` 处于一个中等水平：很强，但并非过度复杂。

**极点质量（0.1275）** 表示这个模式跨层维持的难度。质量越低，传播越自由。`I_0` 相对较
轻，这与它作为广泛存在、持续背景激发的角色相一致。

**位于 π 的动量** 也许是最意外的属性。对 token 方向做傅里叶分解时，`I_0` 在最高频率处
达到峰值（k ≈ π）。这意味着它在相邻 token 之间呈现强烈交替，相当于整个序列上出现一种
“棋盘格”式结构。下面还会回到这个点。

**固定点层（16）和类型（crossover）** 则意味着，直到网络末端，`I_0` 仍未真正进入稳定的
红外区间。它仍在过渡中，从头到尾都保持着明显的动态性与结构性。

### 次级物种：`I_1` 到 `I_5`

其余五个物种振幅要弱得多，但也更专门化：

| 物种 | 最强任务 | 动量 | 固定点类型 | 振幅 |
|---|---|---|---|---|
| `I_1` | logical reasoning | ~0 | IR | 204.3 |
| `I_2` | arithmetic | ~0 | IR | 196.1 |
| `I_3` | syntactic agreement | ~0 | IR | 188.5 |
| `I_4` | pronoun tracking | ~0 | IR | 178.2 |
| `I_5` | factual recall | ~0 | IR | 165.4 |

这里有几点很突出。第一，每个次级物种都最强地对应一种 **不同的推理任务**。它并不是手工
标注出来的分类，而是由“哪类提示词最强地激活哪个模式”自动涌现出的结果。物种与任务类型
之间近似一一对应，本身就是非常有吸引力的结构现象。

第二，所有次级物种都在 **接近零的动量** 处达到峰值。在 token 方向的傅里叶分解里，k ≈ 0
对应随位置缓慢变化、全局存在的模式，这与在 k ≈ π 上交替振荡的骨干模态 `I_0` 恰好相反。

第三，所有次级物种都被标记为 **IR**，也就是它们在后期层已经进入稳定、慢变化的配置。相
比之下，`I_0` 还停留在 crossover 区间。

---

## 分裂的动量结构

`I_0` 位于高动量（k ≈ π），而 `I_1` 到 `I_5` 位于低动量（k ≈ 0），这是数据里最有意思的
结构特征之一。

在 token 格点图像下：

- **k ≈ 0 的模式** 随 token 位置变化缓慢，代表整个输入序列上较均匀、较全局的结构；
- **k ≈ π 的模式** 则是最大程度的交替，类似一种“反铁磁”图样，相邻 token 贡献相位相反。

也就是说，模型似乎同时维持着两种性质截然不同的集体激发：

1. 一个覆盖整个序列、非常强的交替骨干模态；
2. 一组更弱但更有语义局部性的任务敏感模态。

这种分裂是否揭示了变换器如何沿 token 位置组织信息的深层规律，仍然是开放问题。但至少
在这份数据里，它是一个可重复出现的发现。

![Qwen3-4B-Base momentum and helicity]({{ "/assets/images/Qwen3-4B-Base/momentum_helicity_(pronoun_tracking).png" | relative_url }})
*Qwen3-4B-Base 在代词跟踪任务上的动量与 helicity 结构。*

---

## RG 视角：模型在什么地方“稳定下来”

重整化群（RG）分析把模型层数视作一种尺度变换。早层更像“紫外”（细粒度、高频），晚层更
像“红外”（粗粒度、趋于稳定）。

对于 `Qwen3-4B-Base`，大多数次级物种（`I_1` 到 `I_5`）在 **第 13 层左右** 达到固定点，
而 `I_0` 的固定点在 **第 16 层**。

如果用普通神经网络语言来讲，就是：模型在中后段逐渐形成稳定的高层表示。次级、专门化的
模式稍早组织完成；而主导骨干模态则更晚才接近稳定，一直保持动态活性。

下面这些 EFT / RG 图展示了运行质量（模式“权重”随层如何变化）、beta 函数（变化率）以及
相变图景。

![Qwen3-4B-Base RG flow]({{ "/assets/images/Qwen3-4B-Base/rg_flow.png" | relative_url }})
*Qwen3-4B-Base 的 RG 流，展示各 Intelliton 物种如何随网络深度演化。*

![Qwen3-4B-Base EFT parameters]({{ "/assets/images/Qwen3-4B-Base/eft_parameters.png" | relative_url }})
*从 Qwen3-4B-Base 提取出的有效场论参数。*

---

## 所有物种都属于“中质量”

目录里还有一个有趣之处：**六个物种全部被归为中质量**。既没有特别轻、几乎随处可传播的，
也没有特别重、迅速衰减的。

这意味着 `Qwen3-4B-Base` 的主导集体模式处在一个相对窄的动力学带宽中。它们足够稳定，能
跨层、跨提示词重复出现；但又保留足够弹性，能够参与多样任务。换句话说，这个模型似乎是
围绕一组中等稳健的激发组织起来的，而不是围绕一套僵硬冻结的结构。

![Qwen3-4B-Base mass spectrum]({{ "/assets/images/Qwen3-4B-Base/mass_spectrum_(pronoun_tracking).png" | relative_url }})
*Qwen3-4B-Base 在代词跟踪任务上的质量谱。*

---

## 色散关系：模式如何传播

色散关系描述的是，一个模式的能量（或有效频率）如何随动量变化。在自由场理论里，它常常
是一条简单抛物线；在更复杂系统里，形状则会丰富得多。

对于 `Qwen3-4B-Base`，色散关系表现出典型的离散格点模式行为，而主导模式沿着一条很有特征
的曲线分布，进一步确认它确实属于高动量模式。

![Qwen3-4B-Base dispersion relation]({{ "/assets/images/Qwen3-4B-Base/dispersion_relation.png" | relative_url }})
*Qwen3-4B-Base 的格点色散关系。*

---

## 自旋谱：不同任务的内部复杂度

类自旋分数衡量每个模式内部自由度的复杂程度。跨任务观察自旋谱，可以看出哪些任务需要更
复杂的内部激发。

在代词跟踪提示词上，`Qwen3-4B-Base` 的自旋谱在 `I_0` 位置出现明显峰值，而且复杂度处于中
等区间，这与骨干模态保持适度但稳定的内部结构是一致的。

![Qwen3-4B-Base spin spectrum]({{ "/assets/images/Qwen3-4B-Base/spin_spectrum_(pronoun_tracking).png" | relative_url }})
*Qwen3-4B-Base 在代词跟踪任务上的类自旋分数谱。*

---

## 相变：网络内部的重组节点

相变分析会寻找那些激活结构发生定性变化的层，也就是有效自由度发生重组的位置。

对 `Qwen3-4B-Base` 而言，相变图确认了主要重组出现在中后层（大约第 13 到 16 层），与 RG
固定点分析高度一致。

![Qwen3-4B-Base phase transitions]({{ "/assets/images/Qwen3-4B-Base/phase_transitions.png" | relative_url }})
*Qwen3-4B-Base 的相变图。*

---

## 用直白语言总结这张目录

把上面的结果放在一起，`Qwen3-4B-Base` 的 Intelliton 目录讲出了一个相当连贯的故事：

1. 模型维持着一个非常强、在整个序列上交替振荡的骨干激发 `I_0`，几乎对所有任务都活跃；
2. 在这条骨干之上，模型再运行五个更弱、也更任务专门化的模式，它们分别大致对应不同类型
   的推理；
3. 任务专门化模式更早组织完成（约第 13 层），而骨干模式则更晚才接近稳定（约第 16 层）；
4. 所有模式都属于“中质量”，也就是稳定但不僵硬；
5. 主导模式在 token 方向表现为高频交替，任务专门化模式则更偏向全局均匀。

对于一个拥有 40 亿参数的系统来说，这是一个相当有结构的内部图景。下一篇文章会展示：当
模型扩大到 8B，或者经过指令微调之后，这种结构会发生怎样的变化。

---

## 延伸阅读

- 继续阅读 [第 3 篇：用 Intelliton 视角看规模扩展与对齐]({% post_url 2026-04-03-scaling-alignment-intellitons %})
- 继续阅读 [第 4 篇：把幻觉理解为内部不稳定性]({% post_url 2026-04-04-intellitons-and-hallucination %})

</div>