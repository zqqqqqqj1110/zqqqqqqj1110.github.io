---
title: "算法岗八股题目导览"
description: "大模型算法岗面试八股导航：Transformer、GRPO、Workflow & Agent Harness、Agent 范式、RAG 与 BGE-M3、SFT 与 LoRA，手撕代码，以及项目（Agent 1～3），每块附答案链接。"
cover: /images/guide-cover.png
listing:
  priority: 1
date: 2026-10-21 23:57:05
categories:
  - 面试
  - 八股
tags:
  - LLM
---

# 算法岗八股题目导览

> 第 1 部分题目整理自博客园 moonout

目录

- 1\. Transformer 与模型架构原理
- 2\. GRPO 理论（答案：[GRPO理论](/2026/11/01/GRPO%E7%90%86%E8%AE%BA/)）
- 3\. Workflow & Agent Harness（答案：[Workflow与Agent Harness](/2026/11/02/Workflow%E4%B8%8EAgent%20Harness/)）
- 4\. Agent 三大范式（ReAct / Plan-and-Execute / Reflection）（答案：[Agent三大范式与ReAct](/2026/11/02/Agent%E4%B8%89%E5%A4%A7%E8%8C%83%E5%BC%8F%E4%B8%8EReAct/)）
- 5\. RAG 与 BGE-M3（答案：[RAG与BGE-M3](/2026/11/02/RAG%E4%B8%8EBGE-M3/)）
- 6\. SFT 与 LoRA（答案：[SFT与LoRA](/2026/11/02/SFT%E4%B8%8ELoRA/)）
- 7\. LLM 相关代码实现（手撕）
- 8\. 项目（总览：[文档助手](/2026/10/27/%E6%96%87%E6%A1%A3%E5%8A%A9%E6%89%8B/)）

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

### 8. 项目

> 项目总览见：[文档助手](/2026/10/27/%E6%96%87%E6%A1%A3%E5%8A%A9%E6%89%8B/)

#### Agent 1（答案：[Agent1面试题答案](/2026/11/09/Agent1%E9%9D%A2%E8%AF%95%E9%A2%98%E7%AD%94%E6%A1%88/)）

- 请完整介绍 Agent 1 的整体架构、输入输出、核心流程，以及为什么需要使用 Agent。
- 你设计了哪些解析质量指标？这些指标如何计算？如何判断解析结果是否合格？
- 为什么选择 ReAct，而不是单次 LLM 工具路由或 if/else 规则路由？
- Agent 如何选择下一个工具？Observation 包含什么？如何设计停止条件和兜底策略？
- ReAct 中的 Thought、Action、Observation 分别是什么？你的 Agent 1 是如何实现完整循环的？
- 对于扫描 PDF、双栏 PDF、复杂表格、公式混排等难样本，你如何保证解析质量？
- 如果解析工具返回乱码、漏页、超时或错误结果，你如何检测、重试并进行错误归因？
- 什么是轨迹蒸馏？为什么使用 Gemini 作为教师模型？如何构造 Agent 1 的 SFT 数据？
- 什么是拒绝采样？你如何筛选教师模型的成功轨迹？为什么拒绝采样会带来数据偏差？
- 你提到事后纠正和难样本重采样，具体怎么实现？如何验证这些方法有效？
- 为什么要进行 SFT 轨迹蒸馏？相比直接调用教师模型，训练自己的 27B 模型有什么价值？
- LoRA 的原理是什么？你的训练数据如何组织？如何训练模型正确输出工具调用？
- 你项目中的 F1 是如何定义和计算的？Baseline、教师模型和蒸馏模型如何保证公平对比？
- 如何证明轨迹蒸馏和 ReAct 真正有效？你会如何设计消融实验？
- 如果需要处理十万份不同类型的文档，你会如何优化 Agent 1 的延迟、Token 消耗、稳定性和吞吐量？

#### Agent 2（答案：[Agent2面试题答案](/2026/11/09/Agent2%E9%9D%A2%E8%AF%95%E9%A2%98%E7%AD%94%E6%A1%88/)）

- 请介绍 Agent 2 的作用、输入输出以及完整工作流程。为什么不直接使用固定长度切分？
- 为什么选择约 4096 Token 作为 Chunk 大小？Chunk Size 和 Overlap 如何确定？
- 为什么 Agent 2 需要采用 Agent 或模型驱动方案，而不是 RecursiveCharacterTextSplitter 等传统切分方法？
- 如何保证每个 Chunk 的语义完整性？遇到段落、表格、公式和标题跨越 Chunk 边界怎么办？
- 你如何保留文档的 TOC 层级关系？如何解决跨 Chunk 的上下文关联与信息丢失？
- 什么是 GRPO？它与 SFT、PPO 的核心区别是什么？为什么 Agent 2 选择 GRPO？
- GRPO 中 Group、Reward、Advantage 分别是什么？为什么不需要单独训练 Critic？
- GRPO 的 Importance Sampling Ratio、Clipped Objective 和 KL 惩罚分别起什么作用？
- Agent 2 的奖励函数是如何设计的？各项 Reward 分别衡量什么？权重如何确定？
- 如何避免 Reward Hacking？如果语义完整性与 Chunk 长度限制发生冲突，你如何处理？
- 你如何构造 GRPO 训练样本、生成多个候选切分方案、计算奖励并更新模型？
- 你简历中无效 Chunk 比例从 70% 以上降至 0，这里的无效 Chunk 如何定义和统计？
- 如何对比固定长度切分、语义切分、SFT 与 GRPO？你会设计哪些消融实验？
- 如果遇到超长文档、复杂表格、公式、代码块或跨页段落，Agent 2 如何处理？
- 如果 Agent 3 发现知识图谱缺失某些关系，如何判断是 Agent 2 切分导致的？如何定位并修复？

#### Agent 3（答案：[Agent3面试题答案](/2026/11/09/Agent3%E9%9D%A2%E8%AF%95%E9%A2%98%E7%AD%94%E6%A1%88/)）

- 请介绍 Agent 3 的核心任务、输入输出和完整工作流程。它与 Agent 2 的关系是什么？
- 为什么要将文档转换成知识图谱？相比直接使用文本块进行检索或问答，知识图谱能解决什么问题？
- 如何从文本块中抽取实体和实体之间的关系？你的抽取流程是什么？
- 如何定义实体类型和关系类型？如何避免大模型生成不符合要求的实体或关系？
- 如果文本中存在代词、省略、同名实体或者隐含关系，你如何处理？
- 你如何将抽取结果转换成知识图谱？节点、边及其属性分别如何设计？
- 不同文本块可能抽取出同一个实体，你如何进行实体对齐、去重和消歧？
- 如果一个实体关系跨越多个文本块，你如何发现并建立跨块关系？如何避免错误连接？
- 你如何设计实体与关系抽取的提示词？如何约束大模型输出结构化结果？
- 为什么选择当前的模型抽取方案？是否考虑过传统命名实体识别、关系抽取模型或者微调方案？
- 如何评估知识图谱的构建质量？实体准确率、关系准确率、召回率和 F1 分别如何计算？
- 如果大模型抽取出了原文不存在的关系，你如何检测和减少这种幻觉？
- 你如何设计实验验证 Agent 3 的有效性？会选择哪些基线方法和消融实验？
- 如果文档很长，包含大量重复实体和复杂关系，你如何控制图谱规模、构建成本和处理耗时？
- 如果 Agent 4 检查发现知识图谱缺少关系或者出现错误，你如何区分是 Agent 1 解析、Agent 2 切分还是 Agent 3 抽取的问题？
