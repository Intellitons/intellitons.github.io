---
layout: post
title: "What Are Intellitons? A Friendly Guide to the Lattice-Field View"
title_zh: "什么是 Intelliton？一篇看懂晶格场论视角的入门文"
date: 2026-04-01
categories: [introduction, theory]
excerpt: >
  Intellitons are not claims that language models literally contain particles. They are a practical
  way to redescribe the residual stream using a lattice-field coordinate system that makes recurring
  modes easier to see, compare, and talk about.
excerpt_zh: >
  Intelliton 不是说语言模型里真的有粒子，而是把残差流换到一套晶格场论式的坐标系里重新
  描述，好让那些反复出现、能跨层传播的模式更容易被看见、比较和解释。
---

<div data-lang="en" markdown="1">

## Start with the least mysterious version

The Intelliton project is **not** claiming that a language model secretly contains real physical
particles.

The core idea is simpler and more useful than that: take the transformer residual stream, write it
in a coordinate system that physicists already know how to reason about, and ask whether stable,
recurrent modes appear.

At one layer, the residual stream is just a matrix:

- `T` rows for token positions
- `D` columns for hidden channels

You can think of it as a long row of sensors. Each token position has thousands of readings. The
question is not whether any single neuron matters, but whether the whole pattern can be compressed
into a small set of reusable modes.

---

## The sensor analogy

Imagine a sentence with 20 token positions. At each position, instead of one reading, you have a
vector with thousands of numbers. That is what one layer of the residual stream looks like.

Now ask four very ordinary questions:

1. Along the token axis, is the pattern smooth or rapidly oscillating?
2. Inside hidden channels, does it point mostly in one direction or is it a messy mixture?
3. As layers get deeper, does the pattern survive or die out quickly?
4. Does the pattern's internal structure stay tied to a preferred propagation direction?

Those four questions become the project's four main diagnostics:

- **Momentum** answers question 1.
- **Spin-like complexity** answers question 2.
- **Mass** answers question 3.
- **Helicity proxy** answers question 4.

This is why the physics language is useful. It gives a compact way to talk about four different
facets of the same hidden pattern.

---

## A new coordinate system, not a new ontology

The project rewrites the residual stream in a very specific way:

- the **token axis** is treated like a one-dimensional lattice in space,
- the **layer axis** is treated like discrete time,
- the **hidden channels** are treated like internal degrees of freedom.

That mapping is the whole point. It does not say that text is literally matter. It says that a
familiar toolkit from lattice field theory can be borrowed to organise activation patterns.

In code, the main definitions live in `src/lattice_field.py`, and the overall orchestration sits in
`src/intelliton_analyzer.py`.

---

## What momentum means here

Momentum in this project is just a Fourier description of how a mode varies across token positions.

- If the dominant momentum is near `k = 0`, the pattern is broad and smooth across the sequence.
- If the dominant momentum is large, the pattern flips more sharply from one token to the next.

An everyday analogy is an audio equalizer:

- low frequency means slow, smooth variation,
- high frequency means fast, jagged variation.

So when a report says a mode is low-momentum, it is usually saying: this is a sequence-scale pattern,
not a tiny local blip tied to one token.

---

## What spin-like complexity means here

This is the term most likely to confuse readers, because it is **not** literal particle spin.

The project uses SVD to split a layer into dominant modes. In plain language, SVD asks:

> Can this complicated activation matrix be explained mostly by one or two big patterns, or do we
> need many equally important patterns?

If one mode dominates, the internal structure is simple and concentrated. If energy is spread across
many directions, the structure is more mixed and complex. The blog and code call that a spin-like
quantity, but the safer mental model is simply **internal complexity**.

---

## What mass means here

Mass is the most intuitive part of the analogy.

The layer axis is treated like discrete time, and the analysis tracks whether a mode's strength
fades quickly or persists through many layers.

- a **light** mode survives for a long depth range,
- a **heavy** mode dies out quickly.

So mass in this framework is really a measure of **how easily a pattern propagates through the
network**, not how much it weighs in any everyday sense.

---

## What helicity means here

Helicity is also a proxy, not a literal high-energy-physics observable.

The simplified question is: if a mode has a preferred direction on the token lattice, does its
internal structure stay aligned with that direction across layers?

If yes, the mode has a more stable directional signature. If not, the mode is being scrambled.

This is useful because two modes can have similar amplitude but very different directional stability.

---

## Why this framing helps

Once the residual stream is written this way, the project can ask practical questions that are hard
to state cleanly in raw neuron space:

- Which patterns are global versus local across the sequence?
- Which patterns are internally simple versus heavily mixed?
- Which patterns are shallow noise versus deep, persistent carriers?
- Which patterns stay stable across prompts, tasks, and generation steps?

That is the value of Intellitons. They are a compact language for recurring activation patterns.
They are useful if they organise observations better than a giant pile of raw activations.

---

## The shortest correct summary

If you want the plainest possible version, it is this:

> Intellitons are recurring residual-stream modes described in a physics-inspired coordinate system.
> DFT tells you how they vary across tokens, SVD tells you how internally concentrated they are,
> propagator decay tells you how far they travel across layers, and helicity tells you whether their
> internal structure keeps a stable directional signature.

The next article makes that concrete by showing how to read a spectrum report and what `I_0` to
`I_4` sound like in ordinary language.

---

## Continue reading

- [How to Read `I_0` to `I_4`]({% post_url 2026-04-02-inside-qwen-intelliton-spectrum %})
- [Why Different Prompts Light Up Different Intellitons]({% post_url 2026-04-05-why-different-prompts-light-up-different-intellitons %})

</div>

<div data-lang="zh" markdown="1">

## 先从最不神秘的版本开始

Intelliton 项目**不是**在说语言模型里真的藏着物理粒子。

更准确、更实用的说法是：把变换器的残差流换到一套物理学家已经很熟悉的坐标系里，再去看
里面会不会出现稳定、反复出现、可以跨层追踪的模式。

对某一层来说，残差流不过是一个矩阵：

- 行数 `T` 表示 token 位置
- 列数 `D` 表示 hidden channels

你可以把它想成一排传感器。每个 token 位置上都有成千上万个读数。项目真正关心的，不是某
一个神经元是否重要，而是整块信号能不能被少数几个可重复使用的主模式概括出来。

---

## 最通俗的类比：一排传感器

想象一句话有 20 个 token 位置。每个位置上不是一个数字，而是一整个上千维的读数向量。这
就是某一层残差流的大致样子。

现在问四个很朴素的问题：

1. 沿着 token 轴，这个模式是平滑变化，还是快速振荡？
2. 在 hidden channels 里，它更像单一方向，还是复杂混合？
3. 随着层数加深，它能持续很久，还是很快消失？
4. 它的内部结构，是否一直和某个传播方向绑定在一起？

这四个问题，正好对应项目里的四个主诊断量：

- **动量** 对应第 1 个问题
- **类自旋复杂度** 对应第 2 个问题
- **质量** 对应第 3 个问题
- **螺旋度代理量** 对应第 4 个问题

这就是为什么物理语言在这里有用。它把同一批隐藏模式的四个不同侧面，用一套紧凑的词汇串
了起来。

---

## 这是一套新坐标系，不是一套新本体论

项目把残差流这样重写：

- **token 轴** 看成一维晶格上的空间
- **layer 轴** 看成离散时间
- **hidden channels** 看成内部自由度

重点就在这一步。它不是说文本真的变成了物质，而是说可以借用晶格场论里熟悉的工具，来整
理模型内部的激活模式。

在代码里，主要定义集中在 `src/lattice_field.py`，总流程由 `src/intelliton_analyzer.py`
串起来。

---

## 这里的“动量”到底是什么意思

在这个项目里，动量只是描述模式沿 token 位置如何变化的一种傅里叶坐标。

- 如果主导动量接近 `k = 0`，说明这个模式在整个序列上比较平滑、比较全局。
- 如果主导动量较大，说明它在相邻 token 之间切换更快、振荡更强。

最容易懂的比喻是音频均衡器：

- 低频意味着缓慢、平滑的变化
- 高频意味着尖锐、快速的起伏

所以当报告说某个模式是低动量，它通常不是在说“速度慢”，而是在说：这更像一个覆盖整段序
列的大尺度模式，而不是绑在某个 token 上的小噪声。

---

## 这里的“自旋”为什么其实是在看复杂度

这个词最容易让人误会，因为它**不是**粒子物理里的严格自旋。

项目用 SVD 把某一层拆成若干个主模式。人话版的问题其实是：

> 这一层看起来很复杂，但它是不是主要由一两个大模式支配，还是说必须靠很多差不多重要的
> 模式一起才能解释？

如果一个模式特别突出，说明内部结构更集中、更简单。如果能量分散在许多方向上，说明内部
结构更混合、更复杂。博客和代码把这个量借用物理语言叫成 spin-like，但更稳妥的理解就是
**内部复杂度**。

---

## 这里的“质量”为什么就是跨层能活多久

质量是整套类比里最直观的一步。

项目把 layer 轴当成离散时间，然后看一个模式的强度会不会在更深的层里迅速衰减。

- **轻模式** 能持续很多层
- **重模式** 很快就消失

所以这里的质量，实质上是在衡量一个模式**穿透网络深度的能力**，而不是日常意义上的“有多
重”。

---

## 这里的“螺旋度”为什么只是方向性代理量

螺旋度在这里也只是代理量，不是高能物理里那种严格可观测量。

更简单的问法是：如果某个模式在 token 晶格上有偏好的传播方向，它的内部结构会不会在跨层
传播时一直和这个方向绑在一起？

如果会，说明这个模式的方向性签名更稳定。如果不会，说明它在层间被打散了。

这很有用，因为两个模式即使振幅差不多，方向稳定性也可能完全不同。

---

## 为什么这套说法有帮助

一旦把残差流写成这种形式，项目就能提出一些用原始神经元空间很难直接说清的问题：

- 哪些模式是全局的，哪些更局部？
- 哪些模式内部很集中，哪些高度混合？
- 哪些模式只是浅层噪声，哪些能一路传到深层？
- 哪些模式能跨提示词、跨任务、跨生成步骤保持稳定？

Intelliton 的价值就在这里。它提供了一套压缩语言，去描述那些反复出现的激活模式。只要这套
语言比一大堆原始激活更能组织观察结果，它就是有用的。

---

## 一句话总结这件事

如果只保留最通俗也最准确的一句话，那就是：

> Intelliton 是用物理启发坐标系描述出来的残差流重复模式。DFT 看它沿 token 怎么变化，SVD
> 看它内部是否集中，传播子衰减看它能走多深，螺旋度看它的内部结构是否保留稳定的方向性。

下一篇文章会把这件事落到更具体的谱表上，直接教你怎么看 `I_0` 到 `I_4`。

---

## 继续阅读

- [怎么看 `I_0` 到 `I_4`]({% post_url 2026-04-02-inside-qwen-intelliton-spectrum %})
- [为什么不同提示词会点亮不同 Intelliton 模式]({% post_url 2026-04-05-why-different-prompts-light-up-different-intellitons %})

</div>