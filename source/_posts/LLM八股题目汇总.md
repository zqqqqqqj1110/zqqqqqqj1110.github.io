---
title: "LLM 算法岗 | 面试八股题目汇总"
date: 2026-10-21 23:57:05
categories:
  - 面试
  - 八股
tags:
  - LLM
---

# LLM 算法岗 | 面试八股题目汇总

> 第 1 部分题目整理自博客园 moonout

目录

- 1\. Transformer 与模型架构原理
- 2\. GRPO 理论（答案：[GRPO理论](/2026/11/01/GRPO%E7%90%86%E8%AE%BA/)）
- 3\. Workflow & Agent Harness（答案：[Workflow与Agent Harness](/2026/11/02/Workflow%E4%B8%8EAgent%20Harness/)）
- 4\. Agent 三大范式（ReAct / Plan-and-Execute / Reflection）（答案：[Agent三大范式与ReAct](/2026/11/02/Agent%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F%E4%B8%8EReAct/)）
- 5\. RAG 与 BGE-M3（答案：[RAG与BGE-M3](/2026/11/02/RAG%E4%B8%8EBGE-M3/)）
- 6\. SFT 与 LoRA（答案：[SFT与LoRA](/2026/11/02/SFT%E4%B8%8ELoRA/)）
- 7\. LLM 相关代码实现（手撕）

------

### 1. Transformer 与模型架构原理

- 请介绍 Transformer 的结构组成、各部分作用及底层原理。
- Transformer 的 forward 计算包含哪些部件？非线性由什么提供？
- Transformer 为什么能替代 RNN？核心优势是什么？
- 详细介绍 Self-Attention 机制，包括本质、数学解释、具体计算步骤，及时间复杂度。
- 为什么要用 Multi-Head Attention？切分为多头的作用是什么？
- 计算 Attention 的 Softmax 之前为什么要除以根号 $d_k$？
- 介绍一下 Transformer 的位置编码（Positional Encoding），还了解其他位置编码吗？
- 介绍 QKV 的计算。如果在 Transformer 中去掉 K，变成 QQV，会有什么问题？（仅考虑编码器内部）
- Transformer 是 Encoder-Decoder 架构，而 GPT 是 Decoder-only 架构，为什么会演变成这种形式？为什么生成式任务（如 GPT）通常舍弃 Encoder？
- Transformer 的 FFN 层为什么会逐渐演变成 MOE（Mixture of Experts）层？
- MOE 层的负载均衡具体是怎么做的？偏置项 b 是怎么训练的？如何保证偏置项得到变换？
- 如何降低 Transformer 的计算复杂度？为降低计算复杂度，常见的稀疏注意力变体有哪些？
- 分析一下 Transformer 训练过程中显存占用和计算复杂度。
- Self-Attention 机制在多模态对齐上是否存在瓶颈？有没有实际场景里注意力权重完全偏掉的情况？
- 如何解决梯度消失和梯度爆炸问题？
- 介绍一下 LayerNorm 和 BatchNorm 的区别？
- 在 Agent 多轮对话任务中，Attention 的局限性体现在哪些方面？
- 大模型经常一张显卡放不下。为了解决此问题，有哪些模型并行策略？

### 2. GRPO 理论

> 答案见：[GRPO理论](/2026/11/01/GRPO%E7%90%86%E8%AE%BA/)

- 什么是 GRPO？核心思想是什么？
- GRPO 为什么提出？主要解决 PPO 的什么问题？
- GRPO 一次完整训练流程是什么？
- 写出 GRPO 核心目标函数，并解释每项
- Advantage 怎么计算？
- 为什么一个 Prompt 要采样多个 Response？
- GRPO 为什么不需要 Critic？
- PPO 和 GRPO 核心区别？
- Importance sampling ratio 是什么，为什么需要？
- 为什么要 Clip？不 Clip 会怎样？
- Clipped objective 为什么取 min？
- KL Divergence 的作用？
- Reference、Old、Current 三种 Policy 区别？
- Reward 和 Advantage 有什么区别？为何不直接用 Reward？
- 同组 Reward 全相同会怎样？
- Group Size `G` 怎么影响训练？越大越好吗？
- 为什么除以组内标准差，而不只减均值？
- GRPO 是 On-policy 还是 Off-policy？
- GRPO 和 DPO 区别？
- GRPO 和 SFT 的目标区别？
- 为什么 GRPO 常用于数学、代码和推理？
- Sequence-level Reward 怎么作用到 Token？
- KL 和 Clip 都限更新，有何区别？
- GRPO 不稳定有哪些原因？
- Reward Hacking 是什么？为什么发生？
- 怎么判断 GRPO 训练是否有效？
- GRPO 和 DAPO 有什么区别？
- GRPO 和 GSPO 区别？
- Token 级 Clip 与 Sequence 级优化的争议？
- Agentic RL 的过程奖励怎么设计？
- 长 Rollout 影响吞吐，如何优化？
- 手写简化版 GRPO Loss

### 3. Workflow & Agent Harness

> 答案见：[Workflow与Agent Harness](/2026/11/02/Workflow%E4%B8%8EAgent%20Harness/)

- Workflow 是什么？和直接调用一次 LLM 有什么区别？
- Workflow 与自主 Agent 有什么区别？
- 常见 Workflow 编排模式有哪些？
- 怎么设计多步骤 Workflow？如何定义节点之间的输入输出？
- Workflow 为什么需要状态管理？State、Memory、Checkpoint 如何区分？
- 节点失败后如何重试和恢复？
- 如何避免上游错误传递到下游？
- Workflow 如何控制延迟、成本和吞吐？
- 如何评测 Workflow 效果？
- 什么是 Agent Harness？其核心组成是什么？
- Workflow、Agent、Harness 三者是什么关系？
- 结合项目：为什么选择 Workflow，而不是让自主 Agent 自行规划全流程？
- 结合项目：四个处理阶段如何串联？某一阶段出错怎么办？
- 面试官问“为什么用 Workflow 而不是 Harness”，如何纠正并回答？
- 什么时候会从 Workflow 逐步增加 Agent 自主性或 Harness 能力？

### 4. Agent 三大范式（ReAct / Plan-and-Execute / Reflection）

> 答案见：[Agent三大范式与ReAct](/2026/11/02/Agent%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F%E4%B8%8EReAct/)

- 什么是 ReAct？核心思想和完整循环是什么？
- ReAct 的 Thought、Action、Observation 各代表什么？
- ReAct 与 CoT（Chain-of-Thought）有什么区别？
- ReAct 与普通 Function Calling 有什么关系和区别？
- ReAct 如何决定调用什么工具？工具结果如何参与下一步决策？
- ReAct 什么时候停止？怎样避免死循环、重复工具调用和成本失控？
- 如果工具调用失败、超时或返回错误数据，ReAct 应如何处理？
- ReAct 的优缺点是什么？哪些业务适合/不适合？
- ReAct 如何降低幻觉？为什么不能说它能消除幻觉？
- 什么是 Plan-and-Execute？与 ReAct 有什么区别？
- 什么是 Reflection（反思/自检）？它与 ReAct 如何配合？
- ReAct、Plan-and-Execute、Reflection 如何选型？能否组合？
- 你的文档处理项目为什么在局部节点使用 ReAct，而不是纯规则工具路由？
- 既然用了 ReAct，为什么整个系统还要用 Workflow，而不全交给自主 Agent？
- 如果 ReAct 选错了解析工具或产生错误输出，你如何定位、兜底并评测？

### 5. RAG 与 BGE-M3

> 答案见：[RAG与BGE-M3](/2026/11/02/RAG%E4%B8%8EBGE-M3/)

- 什么是 RAG？为什么需要它？
- 请描述一个完整的 RAG 流程。
- RAG 与微调（Fine-tuning）有什么区别？
- 什么是 Embedding？向量检索如何工作？
- 为什么要 Chunking？Chunk 太大或太小有什么问题？
- Top-K 是什么？K 越大越好吗？
- Dense、Sparse 和 Hybrid Retrieval 有什么区别？
- 召回（Retrieval）和重排（Rerank）有什么区别？
- RAG 出现幻觉的原因有哪些？如何缓解？
- RAG 应该怎么评估？
- RAG 中的 Query Rewrite / Multi-query 有什么作用？
- 如果知识库没有相关信息，RAG 应该怎么办？
- BGE-M3 是什么？M3 指什么？
- BGE-M3 的三种检索模式分别是什么？
- BGE-M3 和传统 BM25 有什么区别？
- 用了 BGE-M3，还需要 Reranker 吗？
- 你的多阶段文档处理 Workflow 如果要引入 RAG，会怎么设计？
- 如果面试官问“为什么选 BGE-M3，而不是普通 Embedding 模型？”

### 6. SFT 与 LoRA

> 答案见：[SFT与LoRA](/2026/11/02/SFT%E4%B8%8ELoRA/)

- 什么是 SFT？与预训练有什么区别？
- 从原始数据到 SFT 参数更新的完整流程是什么？
- SFT 的损失函数是什么？为什么是交叉熵？
- Teacher Forcing 是什么？训练与推理有什么区别？
- SFT 中 Prompt 部分是否计算 Loss？如何 Mask？
- SFT 数据质量为什么重要？如何构造和清洗？
- 如何评估 SFT 是否有效？训练 Loss 下降够吗？
- 什么是 LoRA？为什么能减少微调成本？
- 请写出 LoRA 的核心公式并解释维度。
- LoRA 可训练参数量怎么算？举例。
- 为什么低秩分解能起作用？LoRA 的局限性是什么？
- LoRA 的 r、alpha、dropout、target_modules 分别是什么？
- LoRA 为什么常初始化为 A 随机、B 为零？
- LoRA 和全参数微调有什么区别？推理时能否 Merge？
- SFT 和 LoRA 是什么关系？可以同时用吗？
- LoRA 和 QLoRA 的区别是什么？
- SFT 和 DPO、GRPO 有什么区别？
- 如果项目需要小样本定制能力，你会如何选择 SFT + LoRA？

### 7. LLM 相关代码实现（手撕）

- Multi-Head Attention（待补充）
- GRPO（待补充）
