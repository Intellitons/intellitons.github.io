---
layout: post
title: "How to Read `I_0` to `I_4`: A Human Guide to an Intelliton Spectrum"
title_zh: "怎么看 `I_0` 到 `I_4`：一份 Intelliton 谱表的人话读法"
date: 2026-04-02
categories: [case-study, interpretation]
excerpt: >
  Spectrum tables can look intimidating. This article translates a representative Intelliton report
  into ordinary language and explains what `I_0` to `I_4` are doing without over-reading the physics
  metaphor.
excerpt_zh: >
  Intelliton 谱表很容易把人吓住。本文把一份代表性的分析报告翻译成人话，解释 `I_0` 到
  `I_4` 分别像什么，以及动量、类自旋、质量、螺旋度这些词到底该怎么读。
---

<div data-lang="en" markdown="1">

## Read the report with the right mental model first

The safest way to read an Intelliton spectrum is this:

- the species labels `I_0`, `I_1`, `I_2`, and so on are **recurring modes**, not literal particles,
- the physics vocabulary is a compact description of behaviour, not a claim of hidden quantum matter,
- what matters most is the **role** of a mode, not its dramatic name.

In the report discussed here, the main modes are broad sequence-scale patterns rather than tiny
token-local ripples. Their masses are relatively light, which means they persist through many
layers, and their helicity proxy is fairly stable, which means their directional signature is not
being completely scrambled as the network gets deeper.

That already gives a useful picture: these are not one-off flashes. They are reusable internal
carriers.

---

## Before the species list, decode the four columns

If a spectrum table feels abstract, reduce it to four plain questions.

### Momentum

Momentum asks whether the pattern is smooth across token positions or rapidly oscillating.

- low momentum means a broad, global sequence pattern,
- high momentum means sharper token-to-token variation.

In the report discussed here, the important species sit close to the low-momentum end, so the best
mental image is not a tiny local feature but a large-scale background shape spread across the
sequence.

### Spin-like score

This is not literal spin. It is better read as **internal complexity**.

- low spin-like score means one dominant internal direction stands out,
- high spin-like score means several comparable directions are mixed together.

### Mass

Mass tells you how fast a mode fades with depth.

- light modes survive many layers,
- heavy modes disappear quickly.

So when the report says the species are light, it is really saying they are not shallow noise. They
are able to propagate through the stack.

### Helicity proxy

Helicity here means a simplified combination of propagation direction and internal orientation.

- stable helicity means the mode keeps a recognisable directional signature,
- unstable helicity means that signature is getting mixed away.

---

## `I_0`: the default continuation backbone

`I_0` is the easiest species to explain because it is both the strongest and the simplest.

In the report, it has the largest amplitude and the lowest spin-like complexity among the leading
species. The plain-language reading is:

> `I_0` behaves like a strong background mode that helps the model keep a sentence moving toward a
> plausible answer.

It is less like a specific fact and more like a **general continuation scaffold**. When the prompt
is something like "If all dogs are animals..." or "What is 7 + 8?", `I_0` looks like the broad mode
that helps open and stabilise the answer slot.

If you want a slogan, `I_0` is the model's "keep the computation on the rails" mode.

---

## `I_1`: the quiet structural support

`I_1` is best read as a support mode rather than a flashy decision-maker.

In intervention-style reading, changing `I_1` often produces smaller visible output shifts than
changing the stronger causal modes. That does **not** mean it is useless. It usually means it is
too infrastructural to show up as an obvious word swap.

The plain-language reading is:

> `I_1` looks like a structural support mode that helps maintain the shape and stability of the
> representation while other modes do more task-specific work.

Think of it as scaffolding rather than the headline feature.

---

## `I_2`: a reference-resolution mode with a person-like bias

The most intuitive reading of `I_2` comes from pronoun-style prompts such as:

> "Alice gave Bob a book. He thanked her for ..."

In the report, amplifying `I_2` nudges the output toward a more person-centered, masculine pronoun
interpretation. That makes `I_2` feel less like a generic language mode and more like a **reference
selection channel**.

The plain-language reading is:

> `I_2` appears to help the model decide which person-like entity the sentence is currently tracking.

That does not make it a literal "male pronoun particle." It means that, in this probe, the mode is
consistently involved when the model has to collapse a messy discourse context into one concrete
referent.

---

## `I_3`: a higher-complexity mixing mode

`I_3` looks less like a single-purpose button and more like a mixed coordination mode.

Its spin-like complexity is higher, which suggests that it is built from several comparably relevant
internal directions rather than one clean axis. That usually happens in prompts where the model must
hold multiple constraints in mind at once.

The plain-language reading is:

> `I_3` behaves like a mode for combining several partial constraints into one workable internal
> state.

So rather than deciding one token directly, `I_3` is better imagined as part of the middle-layer
machinery that keeps complex reasoning or structured sentence interpretation coherent.

---

## `I_4`: a complementary reference mode

`I_4` looks related to `I_2`, but with a different directional bias in pronoun-style settings.

In the report, amplifying `I_4` can nudge outputs toward forms like "her" rather than a neutral or
object-like continuation. The plain-language reading is:

> `I_4` is another reference-sensitive mode, complementary to `I_2`, and appears when the model has
> to settle on a different discourse framing of who is being talked about.

This is useful because it shows that "pronoun tracking" is not a single monolithic skill. The model
can separate that work into several nearby but distinct modes.

---

## What the whole spectrum says in one paragraph

Taken together, `I_0` to `I_4` tell a coherent story.

- `I_0` is a strong general backbone.
- `I_1` helps keep the internal state stable.
- `I_2` and `I_4` are more obviously tied to reference selection.
- `I_3` looks like a higher-complexity mixing mode.

That is why the Intelliton view can be useful. It turns a huge hidden state into a cast of recurring
roles.

---

## What not to over-interpret

There are two important cautions.

1. Species indices are bookkeeping labels. `I_2` in one run is not guaranteed to mean exactly the
   same thing in every future run.
2. Terms like momentum, spin, and helicity are proxies. They organise evidence, but they are not
   proof that the network literally contains particle-like objects.

The disciplined reading is: these labels help summarise recurrent activation roles.

---

## Continue reading

- [Why Different Prompts Light Up Different Intellitons]({% post_url 2026-04-05-why-different-prompts-light-up-different-intellitons %})
- [Scaling and Alignment Through the Intelliton Lens]({% post_url 2026-04-03-scaling-alignment-intellitons %})

</div>

<div data-lang="zh" markdown="1">

## 先用对心智模型，再看谱表

读 Intelliton 谱表时，最稳妥的起点是这三句话：

- `I_0`、`I_1`、`I_2` 这些名字表示的是**反复出现的模式**，不是字面意义上的粒子
- 物理词汇是对行为的压缩描述，不是说模型里藏着量子物质
- 最重要的不是名字本身，而是每个模式在计算里扮演了什么角色

在这里讨论的这份报告里，几个主导模式更像覆盖整段序列的大尺度结构，而不是只绑在某个
token 上的一次性小波纹。它们的质量都偏轻，说明能跨很多层传播；它们的螺旋度代理量也相
对稳定，说明这种方向性签名没有在层间被完全打散。

这已经很值得注意了：这些模式不是一闪而过的火花，而是可重复使用的内部载体。

---

## 先把四列术语翻译成人话

如果一张谱表看上去很抽象，就先把它压缩成四个问题。

### 动量

动量问的是：这个模式沿 token 位置是平滑的，还是快速振荡的？

- 低动量表示更全局、更平滑的序列模式
- 高动量表示相邻 token 之间变化更快

这份报告里，重要物种都更靠近低动量端，所以更合适的心智图像不是“某个 token 上的小机关”，
而是“覆盖整个序列的大背景形状”。

### 类自旋分数

这不是字面意义上的自旋，更适合读成**内部复杂度**。

- 分数低，说明一个内部方向特别突出
- 分数高，说明多个方向混在一起，结构更复杂

### 质量

质量说的是一个模式会不会随着层数加深而快速衰减。

- 轻模式能活很多层
- 重模式很快消失

所以当报告说这些物种都偏轻，本质意思就是：它们不是浅层噪声，而是能一路传播到更深层的
内部模式。

### 螺旋度代理量

这里的螺旋度，是传播方向和内部朝向结合起来的一个简化指标。

- 稳定说明模式保留了可辨认的方向性签名
- 不稳定说明这种签名被混掉了

---

## `I_0`：最强的默认续写底座

`I_0` 是最容易解释的一个物种，因为它既最强，也最简单。

在这份报告里，它的振幅最大，而且在主导物种里类自旋复杂度最低。最直白的人话解释是：

> `I_0` 很像一个强背景模式，用来保证模型把句子继续往一个合理答案上推进。

它不像某条具体知识，更像一个**通用续写骨架**。当提示词是“如果所有狗都是动物……”或
“7 + 8 等于多少？”这种形式时，`I_0` 看起来像是在把“答案槽位”撑开并稳定住的那股力。

如果硬要压缩成一句话，`I_0` 就像模型里那个“先让计算别跑偏”的总底座。

---

## `I_1`：安静但重要的结构支撑

`I_1` 更适合被读成支撑模式，而不是最显眼的决策按钮。

在干预式阅读里，改动 `I_1` 往往不会像改强因果模式那样，立刻把某个词换掉。这**不**代表它
没用，更常见的解释是：它太基础、太基础设施化了，所以表面输出不一定马上剧烈变化。

更合适的人话解释是：

> `I_1` 像一个维持表示形状和系统稳定性的结构模式，让其他更任务化的模式在上面工作。

它更像脚手架，而不是舞台中央的主角。

---

## `I_2`：带有人物指代偏向的引用解析模式

`I_2` 最好理解的场景，是这类代词提示词：

> “Alice gave Bob a book. He thanked her for ...”

在这份报告里，放大 `I_2` 会把输出往更偏人物、偏男性代词解释的方向推。这让它不像一个通
用语言模式，而更像一条**指代选择通道**。

更通俗地说：

> `I_2` 看起来会帮助模型决定，这句话现在到底在跟踪哪一个“人”。

这并不意味着它是一个字面意义上的“男性代词粒子”。更稳妥的理解是：在这个探针设置里，只
要模型需要把混杂的语篇上下文压缩成一个明确先行词，`I_2` 就会稳定参与进来。

---

## `I_3`：更高复杂度的混合协调模式

`I_3` 不像一个单用途按钮，更像一个负责混合多种约束的协调模式。

它的类自旋复杂度更高，说明它不是沿着一条干净轴工作，而是由几个同样重要的内部方向共同
构成。这往往出现在模型需要同时维持多个约束的提示词里。

更合适的人话解释是：

> `I_3` 像是在把几条半成品约束揉成一个可用内部状态的模式。

所以与其把它想成直接拍板某个 token 的按钮，不如把它想成中间层里保持复杂推理或结构理解
不散架的那台“混合器”。

---

## `I_4`：与 `I_2` 互补的另一条指代模式

`I_4` 和 `I_2` 有相似之处，但在代词类场景里又带着不同的方向偏好。

在这份报告里，放大 `I_4` 会把输出往 `her` 这类形式推，而不是中性或其他续写。更通俗的读
法是：

> `I_4` 也是一个对指代敏感的模式，只是它和 `I_2` 在“当前到底在说谁”这个问题上，代表了
> 不同的语篇落点。

这点很重要，因为它说明“代词跟踪”不是一个整块技能。模型可以把这项工作拆成几条彼此相近、
但又不完全相同的内部模式。

---

## 把整张谱表压成一段话

把 `I_0` 到 `I_4` 合起来看，故事其实很连贯：

- `I_0` 是强而通用的背景底座
- `I_1` 负责稳住结构
- `I_2` 和 `I_4` 更明显地参与指代选择
- `I_3` 更像复杂约束的混合模式

这就是 Intelliton 视角的价值。它把一大片难以直视的隐藏状态，压缩成一组反复出现的“角色分工”。

---

## 哪些地方不要过度解读

这里有两个很重要的保留意见。

1. 物种编号只是记账标签。一次运行里的 `I_2`，不保证永远和另一次运行里的 `I_2` 完全等价。
2. 动量、自旋、螺旋度这些词都是代理量。它们是在组织证据，不是在证明网络里真的有字面意
   义上的粒子。

最稳妥的读法是：这些标签在帮我们总结反复出现的激活角色。

---

## 继续阅读

- [为什么不同提示词会点亮不同 Intelliton 模式]({% post_url 2026-04-05-why-different-prompts-light-up-different-intellitons %})
- [用 Intelliton 视角看规模扩展与对齐]({% post_url 2026-04-03-scaling-alignment-intellitons %})

</div>