---
layout: post
title: "Why Different Prompts Light Up Different Intellitons"
title_zh: "为什么不同提示词会点亮不同 Intelliton 模式"
date: 2026-04-05
categories: [applications, tasks, interpretation]
excerpt: >
  All prompt categories look like next-token prediction from the outside, but inside the model they
  ask for different kinds of work. This article uses the project's five prompt families to explain
  why different Intelliton modes become active.
excerpt_zh: >
  从外面看，五类提示词都像“预测下一个 token”；但从模型内部看，它们要求的工作完全不同。
  本文结合项目里的五类提示词，解释为什么不同 Intelliton 模式会被点亮。
---

<div data-lang="en" markdown="1">

## The same interface can hide very different internal jobs

From the outside, every prompt in this project looks similar: the model reads a prefix and predicts
what comes next.

Inside the model, that similarity is misleading.

The prompt categories in `src/datasets.py` force the network to solve different kinds of internal
problems. That is why they can light up different Intelliton modes even when every task is framed as
plain text continuation.

The key point is simple:

> The output interface is always next-token prediction, but the hidden computation needed to get
> there can be very different.

---

## The five prompt families

The project uses five prompt categories:

1. pronoun tracking
2. factual recall
3. logical reasoning
4. arithmetic
5. syntactic agreement

Each category puts pressure on a different part of the model's internal machinery.

---

## Pronoun tracking: who does "she" refer to?

Example prompts include:

- "Alice gave Bob a book. He thanked her for ..."
- "The teacher asked the student a question. She answered ..."

These prompts are hard because the model has to keep several candidate entities alive at once and
then decide which one the next pronoun should point to.

That means the model must track:

- entity identity,
- gender and number cues,
- discourse role,
- which referent is currently most active.

This is why pronoun-tracking prompts often illuminate reference-sensitive modes. The model is not
just choosing a word. It is doing discourse bookkeeping.

---

## Factual recall: pull a stable answer from memory

Example prompts include:

- "The capital of France is ..."
- "The chemical formula for water is ..."

These are different from pronoun tasks because there is usually one highly preferred answer already
stored in the model's long-range memory.

The main internal job is not to juggle many local candidates, but to retrieve and stabilise a very
high-confidence continuation.

That is why factual recall often looks more robust under small perturbations. A mapping such as
"France -> Paris" is usually supported by several redundant internal routes rather than one fragile
single mode.

---

## Logical reasoning: compress several premises into one conclusion

Example prompts include:

- "If all dogs are animals, and all animals are living things, then all dogs are ..."
- "If A is taller than B, and B is taller than C, then A is ..."

These prompts ask the model to combine multiple statements before it can produce the next token.

So the network needs more than lexical memory. It needs an internal state that keeps the rules,
relations, and target conclusion aligned long enough to land on the right answer.

This is why logical reasoning often co-activates a strong global backbone mode plus one or more
higher-complexity mixing modes.

---

## Arithmetic: build the answer slot, then fill it

Example prompts include:

- "What is 7 + 8? The answer is ..."
- "What is 100 divided by 5? The answer is ..."

Arithmetic resembles logical reasoning in one important way: the answer is not a high-frequency word
you can emit immediately. The model has to transform the prefix into a more structured internal
state first.

That usually means two kinds of work:

- create or stabilise an answer-bearing state,
- carry a small symbolic or numerical transformation.

This is why arithmetic prompts often share some modes with logical reasoning while still showing
their own task-specific preferences.

---

## Syntactic agreement: keep the sentence grammatically on track

Example prompts include:

- "The group of students were studying hard. Each of them was ..."
- "Not only the teacher but also the students were excited about the ..."

These prompts are neither mainly about world knowledge nor mainly about arithmetic.

Their difficulty comes from grammatical structure:

- what is the true syntactic head,
- what number agreement should be maintained,
- what verb form or continuation is locally licensed.

So syntactic-agreement prompts often rely on a broad continuation scaffold plus a more local
structure-sensitive correction signal.

---

## Why similar low-momentum modes can still do different jobs

An easy mistake is to think that if several species sit near low momentum, they must be doing the
same thing.

Not so.

Low momentum only says they are broad sequence-scale patterns rather than sharp token-local ripples.
Two low-momentum modes can still differ in at least three important ways:

1. they can point in different hidden-channel directions,
2. they can have different amplitude and causal strength,
3. they can propagate differently across layers.

So two modes can both be global while still supporting very different kinds of internal work.

---

## A practical reading guide

If you want to read a task-to-mode result quickly, use this checklist.

1. If pronoun prompts are sensitive to a mode, ask whether that mode is helping with referent
   selection.
2. If arithmetic and logical reasoning co-activate a mode, ask whether it is building an abstract
   answer state rather than recalling a memorised phrase.
3. If factual recall stays robust under perturbation, ask whether the knowledge is distributed across
   several redundant routes.
4. If syntactic prompts shift without changing global meaning, ask whether the mode is enforcing a
   grammatical form rather than a semantic fact.

This is how the Intelliton framework becomes useful: it turns prompt categories into hypotheses
about internal computational roles.

---

## The shortest summary

Different prompts light up different Intellitons because they require different hidden work.

- pronoun tracking needs discourse binding,
- factual recall needs stable memory retrieval,
- logical reasoning needs relation composition,
- arithmetic needs symbolic transformation,
- syntactic agreement needs grammatical control.

They all look like next-token prediction from the outside. They do not look the same from inside the
residual stream.

---

## Continue reading

- [How to Read `I_0` to `I_4`]({% post_url 2026-04-02-inside-qwen-intelliton-spectrum %})
- [Hallucination as Internal Instability]({% post_url 2026-04-04-intellitons-and-hallucination %})
- [Refusal as an Intelliton]({% post_url 2026-04-06-refusal-as-intelliton %})

</div>

<div data-lang="zh" markdown="1">

## 外面看都像续写，里面做的却不是同一种活

从外面看，这个项目里的所有提示词都很像：模型读入前缀，然后预测接下来的 token。

但如果往模型内部看，这种相似性其实很有迷惑性。

`src/datasets.py` 里的几类提示词，会迫使网络去解决完全不同的内部问题。这也是为什么它们虽
然都表现成普通的文本续写，却会点亮不同的 Intelliton 模式。

最关键的一句话是：

> 输出接口永远都是 next-token prediction，但为了走到这个输出，模型内部需要完成的计算工作可
> 以很不一样。

---

## 项目里用了五类提示词

项目里的提示词主要分成五类：

1. pronoun tracking
2. factual recall
3. logical reasoning
4. arithmetic
5. syntactic agreement

每一类都在给模型内部的不同部件施加压力。

---

## 代词跟踪：这句里的 “she” 到底指谁？

典型例子包括：

- “Alice gave Bob a book. He thanked her for ...”
- “The teacher asked the student a question. She answered ...”

这类提示词之所以难，是因为模型要同时保留多个候选实体，然后再决定下一个代词到底该指向
哪一个。

这意味着模型必须追踪：

- 实体是谁
- 性别和单复数线索
- 语篇角色
- 当前哪个先行词最活跃

所以代词跟踪任务很容易点亮那些对指代敏感的模式。模型不只是选一个词，它还在做一整套语
篇记账。

---

## 事实回忆：从记忆里拉出一个稳定答案

典型例子包括：

- “The capital of France is ...”
- “The chemical formula for water is ...”

这和代词任务不一样，因为这里通常已经存在一个非常强的候选答案，模型要做的更多是把它从
长期记忆里取出来并稳定住。

核心工作不是在句内多个候选之间来回权衡，而是提取并巩固一个高置信的续写。

这也是为什么事实回忆在小扰动下往往更稳。像 “France -> Paris” 这种映射，通常不是靠一条
脆弱单通道支撑，而是有几条冗余内部路径在共同支持。

---

## 逻辑推理：先把前提揉成结论，再落词

典型例子包括：

- “If all dogs are animals, and all animals are living things, then all dogs are ...”
- “If A is taller than B, and B is taller than C, then A is ...”

这类提示词要求模型在输出下一个 token 之前，先把多条前提组合起来。

所以网络需要的不只是词汇记忆，还需要一种能把规则、关系和目标结论暂时维持在一起的内部
状态，直到答案真正落出来。

这也是为什么逻辑推理经常会同时点亮一个强全局底座模式，再加上一两个更高复杂度的混合模
式。

---

## 算术：先把答案槽位搭起来，再把数值放进去

典型例子包括：

- “What is 7 + 8? The answer is ...”
- “What is 100 divided by 5? The answer is ...”

算术和逻辑推理有一个相同点：答案不是一个能立刻凭语料频率吐出来的高频词，模型往往需要
先把前缀变成更结构化的内部状态。

这通常包含两种工作：

- 建立或稳定一个承载答案的内部状态
- 完成一个小型符号或数值变换

所以算术题常常会和逻辑题共享一部分模式，但同时又保留它自己的任务偏好。

---

## 句法一致性：把句子在语法上维持住

典型例子包括：

- “The group of students were studying hard. Each of them was ...”
- “Not only the teacher but also the students were excited about the ...”

这类提示词的难点，既不主要是世界知识，也不主要是算术，而是句法结构本身：

- 真正的句法中心是谁
- 单复数一致性如何保持
- 当前应该落下哪种词形或续写形式

因此，句法一致性任务通常会依赖一个比较广的续写底座，再加上一条更关注局部结构修正的信
号。

---

## 为什么都是低动量，也完全可能分工不同

一个很容易犯的错误是：如果好几个物种都靠近低动量，那它们是不是就在做同一件事？

并不是。

低动量只说明它们都是覆盖序列尺度的大模式，而不是绑在某个 token 上的小波纹。即便如此，
两个低动量模式仍然可以在至少三点上完全不同：

1. 它们可以指向不同的 hidden-channel 方向
2. 它们的振幅和因果强度可以不同
3. 它们跨层传播的方式可以不同

所以，两个模式都很“全局”，不代表它们的内部工作内容也一样。

---

## 一份实用读法

如果你想快速读懂“任务类型和模式激活”的对应关系，可以用下面这张小清单。

1. 如果代词提示词对某个模式特别敏感，先问它是不是在帮模型做先行词选择。
2. 如果算术和逻辑推理同时点亮某个模式，先问它是不是在构建抽象答案状态，而不只是回忆固定
   短语。
3. 如果事实回忆在扰动下仍然很稳，先问知识是不是被分布在几条冗余通路上。
4. 如果句法任务会变、但全局语义没有变，先问这个模式是不是在约束语法形式，而不是语义事实。

Intelliton 框架的用处就在这里：它把任务类别变成了对内部计算角色的可检验假设。

---

## 最短总结

不同提示词会点亮不同 Intelliton，是因为它们要求模型完成的隐藏工作不同。

- 代词跟踪要做语篇绑定
- 事实回忆要做稳定记忆提取
- 逻辑推理要做关系组合
- 算术要做符号变换
- 句法一致性要做语法控制

从外面看，它们都像 next-token prediction。从残差流内部看，它们一点也不像同一种计算。

---

## 继续阅读

- [怎么看 `I_0` 到 `I_4`]({% post_url 2026-04-02-inside-qwen-intelliton-spectrum %})
- [把幻觉理解为内部不稳定性]({% post_url 2026-04-04-intellitons-and-hallucination %})
- [拒绝即 Intelliton]({% post_url 2026-04-06-refusal-as-intelliton %})

</div>