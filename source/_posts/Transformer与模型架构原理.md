---
title: "Transformer与模型架构原理"
date: 2026-11-25 14:09:15
categories:
  - 面试
  - 八股
tags:
  - Transformer
  - LLM
---

# Transformer 与模型架构原理：LLM 算法岗面试复习笔记

> 基于 MoonOut 在博客园发布的《LLM 算法岗 | 八股问答（1）· Transformer 与模型架构原理》所涉及的 18 个主题，**重新概括、组织并校正**的学习笔记；不是原文逐字转载。
>
> 来源：https://www.cnblogs.com/moonout/p/19702708
>
> 阅读方式：先记每节的“面试回答”，再根据公式与追问补充展开。

## 1. Transformer 架构

**面试回答：**原始 Transformer 是编码器—解码器结构。编码器负责利用双向上下文形成输入表示，解码器使用因果自注意力逐 token 生成，并通过交叉注意力读取编码器表示。每层核心由注意力、位置独立的前馈网络、残差连接与归一化构成。

| 部件 | 职责 |
| --- | --- |
| Token Embedding | 将 token ID 映射为连续向量 |
| Position 信息 | 为注意力模型引入顺序或相对位置 |
| Self-Attention | 聚合序列内相关位置的内容 |
| Cross-Attention | 让目标序列访问源序列表示 |
| FFN / MLP | 对每个 token 的特征做非线性变换 |
| Residual | 保留恒等信息通路，改善优化 |
| LayerNorm / RMSNorm | 规范化激活，提升训练稳定性 |
| LM Head | 将隐藏状态映射到词表 logits |

**注意：**原始 Transformer 使用 Post-Norm；大量现代 LLM 使用 Pre-Norm 或相关变体。

## 2. Forward 与非线性

**面试回答：**一次典型 Transformer block 的前向传播包括归一化、QKV 投影、注意力分数及 Softmax、加权求和、输出投影、残差连接，以及两层或门控 MLP。主要非线性来自 MLP 激活、注意力 Softmax，也包括归一化等操作。

```text
X → Norm → Attention(Q,K,V) → Residual Add
  → Norm → MLP(activation/gating) → Residual Add
```

以上为 **Pre-Norm** 示意。不能说“去掉 FFN 激活整个模型就变成线性网络”，因为 Softmax 等仍是非线性的。

## 3. Transformer 与 RNN

**面试回答：**RNN 的序列位置之间存在递归依赖，训练不易沿时间维并行，远距离信息传播的路径也长。Transformer 可对训练序列的注意力运算进行矩阵并行，任意两个位置可直接交互，因此更利于扩展。不过标准全注意力有二次序列长度成本，自回归推理仍需要逐 token 生成。

- RNN 典型层计算量：$O(nd^2)$（依具体结构而异），时间步串行。
- 标准注意力核心：$O(n^2d)$，且 QKV、输出投影与 MLP 还会产生 $O(nd^2)$ 开销。
- Attention 权重**不等同于可靠的模型解释**。

## 4. Self-Attention

设输入 $X\in\mathbb R^{n\times d}$，单头的投影为

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

注意力输出为

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

其中 $M$ 可以是因果或 padding mask（不允许关注的位置填入很大的负数）。

**面试时按五步讲：**投影 Q/K/V → 计算 $QK^\top$ → 按 $\sqrt{d_k}$ 缩放并施加 mask → 对每个 query 的 key 维度 Softmax → 对 V 加权求和。

- 注意力分数矩阵形状：$n\times n$。
- 计算 $QK^\top$ 与 $AV$：$O(n^2d)$ 量级。
- 显式存储注意力矩阵：每头 $O(n^2)$；FlashAttention 能减少显存读写和显式矩阵存储，但**精确全注意力的算术复杂度仍是二次量级**。

## 5. Multi-Head Attention

**面试回答：**多头把表示映射到不同的可学习子空间，允许模型并行学习不同类型的相关性，然后拼接并通过输出投影融合。

$$
\mathrm{head}_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V),
$$

$$
\operatorname{MHA}(X)=\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O.
$$

通常取 $d_k=d_v=d/h$。在固定总隐藏维度时，多头不意味着注意力复杂度自动降低为 $1/h$：共有 $h$ 个 $n\times n$ 的权重矩阵。每个 head 也不保证一定学会可解释的“语法/语义”分工。

## 6. Attention 缩放

**面试回答：**当 Q、K 各坐标近似独立、零均值且单位方差时，点积 $q\cdot k=\sum_{j=1}^{d_k}q_jk_j$ 的方差约为 $d_k$。维度越大，logits 的尺度越容易过大，使 Softmax 变得尖锐且不易优化。除以 $\sqrt{d_k}$ 可以把点积分数的典型尺度规范到更稳定的范围。

$$
\operatorname{Var}(q\cdot k)\approx d_k,\qquad
\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{d_k}}\right)\approx1.
$$

这里是基于初始化分布的直觉，不是所有训练阶段都严格成立。

## 7. 位置编码

**面试回答：**不考虑 mask 等时，标准自注意力对 token 顺序具有置换等变性，本身不能区分绝对位置，需要显式或隐式的位置机制。

| 方法 | 作用方式 | 面试关键词 |
| --- | --- | --- |
| Sinusoidal | 对输入增加固定正弦余弦向量 | 原始 Transformer、不同频率 |
| Learned Position | 学习位置嵌入表 | 简单、固定训练范围 |
| RoPE | 旋转 Q、K 的各二维子空间 | 相对位置信息、长上下文扩展 |
| ALiBi | 给注意力 logits 加距离偏置 | 不额外学习位置嵌入 |

原始绝对正弦编码：

$$
PE(pos,2i)=\sin\!\left(pos/10000^{2i/d}\right),\quad
PE(pos,2i+1)=\cos\!\left(pos/10000^{2i/d}\right).
$$

**易追问：**RoPE 并不天然保证任意长度可靠外推；长上下文通常还需配套缩放、训练或其他工程策略。

## 8. 为什么 QKV 而非 QQV

**面试回答：**Q 表示“要找什么”，K 表示“有哪些信息可匹配”，V 表示“被读取的内容”。独立投影允许查询与被查询内容处在不同的表示空间。

若改为 QQV：

$$
A=\operatorname{softmax}\left(\frac{QQ^\top}{\sqrt{d_k}}\right),\qquad Z=AV.
$$

此时 **Softmax 前的 $QQ^\top$ 是对称半正定矩阵**，表达约束比一般的 $QK^\top$ 更强；但逐行 Softmax 后 $A$ **不一定对称**。因此准确说法是“限制打分结构与参数自由度”，不能说“注意力权重必然对称”或“完全失去动态性”。

## 9. Encoder-Decoder 与 Decoder-only

**面试回答：**架构选择通常与预训练目标和推理模式有关。编码器适合理解双向上下文；编解码器适合条件生成；Decoder-only 用因果 mask 做下一个 token 预测，可以将问答、摘要、指令等统一为续写式条件生成，并利用 KV cache 加速自回归解码。

| 结构 | 注意力形式 | 常见训练目标/任务 |
| --- | --- | --- |
| Encoder-only | 双向注意力 | 表征学习、分类、MLM（如 BERT） |
| Encoder-Decoder | 编码器双向 + 解码器因果 + cross-attention | 翻译、条件文本生成（如 T5） |
| Decoder-only | 因果自注意力 | 下一 token 预测、通用生成（如 GPT） |

**不要绝对化：**生成任务并非只能使用 Decoder-only；Encoder-Decoder 仍是有效的生成模型范式。

## 10. FFN 与 MoE

**面试回答：**密集 FFN 每个 token 都经过同一组网络。MoE 让 Router 按 token 选择少量专家，使激活参数数量与模型总参数规模部分解耦，从而在给定单 token 算力预算下扩大参数容量。

典型 Top-$k$ MoE：

$$
y=\sum_{i\in\operatorname{TopK}(r(x))}g_i(x)\,E_i(x).
$$

优势包括扩展参数容量；代价包括专家负载不均、跨设备通信、路由稳定性和推理部署复杂度。并不是所有模型都由 FFN 改为 MoE。

## 11. MoE 负载均衡

**面试回答：**Router 容易把 token 集中分配给少数专家，导致拥塞与资源浪费。可使用辅助负载均衡损失、专家容量限制，以及按实际负载动态调节路由偏置等策略。

- **辅助损失方式：**把路由概率/选中频率的均衡目标加入训练损失，通过梯度影响 Router。
- **动态偏置方式：**维护每个专家的路由修正项；过载专家减小偏置，低负载专家增大偏置。某些方案中偏置**不是普通梯度下降参数**，而是根据专家负载使用单独规则更新。
- **面试注意：**“偏置 $b$ 怎么训练”必须先问清具体算法（如 DeepSeek-V3 的辅助损失自由负载均衡方案），不能一概而论。

**补充：DeepSeek-V3 的偏置怎么更新（无辅助损失的负载均衡）**

每个专家 $i$ 维护一个偏置 $b_i$，只用来**选专家**，不参与门控权重：

$$
\text{选择：}\operatorname{TopK}_i\big(s_i(x)+b_i\big),\qquad
\text{加权：}g_i(x)\propto s_i(x).
$$

- **怎么训练**：$b_i$ 不走反向传播。每个训练 step 结束后统计各专家实际接收的 token 数：过载的专家 $b_i \leftarrow b_i-\gamma$，欠载的专家 $b_i \leftarrow b_i+\gamma$（$\gamma$ 为偏置更新速率）。
- **如何保证偏置会变**：它按负载的符号规则直接改写，不依赖梯度，只要负载不均就一定会被调整，不存在“梯度太小学不动”的问题。
- **为什么只加在选择上**：门控权重仍用原始亲和度 $s_i(x)$，偏置只改变“去哪个专家”，不扭曲专家输出的加权方式，也避免了辅助损失干扰主任务梯度。DeepSeek-V3 另外保留了一个系数很小的序列级平衡损失，防止单条序列内极端不均。

## 12. 降低注意力复杂度

**面试回答：**分为两类：减少 token 两两交互，或近似原本的注意力核；此外还有改善显存与访存而不改变全注意力算术阶数的工程优化。

| 方法 | 思路 |
| --- | --- |
| Sliding Window / Local Attention | 每个 token 只看附近窗口，近似 $O(nwd)$ |
| Global + Local / Block Sparse | 部分 token 具有全局连边，其他只保留稀疏连接 |
| LSH / Reformer | 哈希分桶，优先计算可能相关的 token 对 |
| Low-rank / Linformer | 压缩部分序列维度 |
| Kernelized / Performer | 用特征映射近似 Softmax attention |
| FlashAttention | 精确 attention 的 IO-aware 计算，显著优化显存使用与吞吐，但不把 $O(n^2d)$ 算术直接改成线性 |

## 13. 训练显存与 FLOPs

**面试回答：**训练显存拆成参数、梯度、优化器状态、激活缓存与临时工作区。复杂度同时受序列长度 $n$、模型宽度 $d$、层数 $L$、batch size $B$ 影响。

常规 dense block 的算术量级：

$$
\mathrm{FLOPs}_{\mathrm{forward}} = O\!\left(BL(nd^2+n^2d)\right).
$$

对 Adam 类训练，若采用混合精度与 FP32 优化器状态，单参数的内存可粗略拆为：模型参数、梯度、主权重副本以及一阶/二阶矩，具体字节数取决于精度、优化器和分片方案，**不能把 16 bytes/param 当成普适常数**。

- 显式注意力权重缓存：$O(BLhn^2)$（$h$ 是头数）。
- FFN 激活：规模受 $BLnd_{ff}$ 影响。
- 常用省显存手段：activation checkpointing、FlashAttention、ZeRO/FSDP、混合精度、低精度优化器、并行分片。
- 计算量与显存占用是两个不同问题，必须分别回答。

**补充：可以直接说出口的估算**

- **静态显存（混合精度 + Adam 的常见配置）**：BF16 参数 2 + BF16 梯度 2 + FP32 主权重 4 + Adam 一阶矩 4 + 二阶矩 4 ≈ **16 字节/参数**。7B 模型约 $7\times10^9\times16\approx112$ GB，还没算激活，单卡 80GB 放不下全参训练，需要 ZeRO/FSDP 分片，或改用 LoRA/QLoRA。
- **训练计算量**：经验公式 $C\approx 6ND$（$N$ 参数量，$D$ 训练 token 数），前向约 $2ND$、反向约 $4ND$；序列很长时还要加上注意力的 $O(n^2d)$ 项。
- **激活显存**：随 $B\cdot n\cdot L\cdot d$ 线性增长；不用 FlashAttention 时，注意力矩阵还会再加一项 $B\cdot h\cdot n^2\cdot L$。长序列训练时激活往往比参数更占显存，这正是 activation checkpointing 的用武之地。

## 14. 多模态对齐

**面试回答：**文本 token、视觉 patch、音频帧在特征分布、时间/空间尺度和语义粒度上不同，仅拼接后注意力交互不必然产生精确的跨模态对齐。还需要合适的投影模块、训练目标和细粒度监督。

- **主要瓶颈：**模态间分布差异；局部图像区域和词语的对应关系；长视觉序列的开销；语言先验压过视觉证据。
- **常见策略：**跨模态投影、cross-attention、对比学习、图文匹配、指令微调、区域级对齐。
- **谨慎表述：**“模型生成错了”并不能直接证明“某个注意力头的权重偏了”。若要提出具体案例，需要可复现的注意力可视化或消融实验证据。

**补充：可以举的实际现象**

- **物体幻觉**：问 VLM“图里有没有狗”，模型受语言先验影响直接回答“有”，对图像 token 的注意力占比很低。常见的观察是：生成越往后，对视觉 token 的注意力越少，越依赖已生成的文本，幻觉也越多。
- **Attention sink**：大量注意力集中在第一个 token 或少数几个“无信息”的视觉 token 上，真正相关的图像区域反而只分到很少的权重。
- **细粒度定位失败**：计数、读小字、区分左右位置这类任务，需要精确对应到局部 patch，而全局图文对齐目标（如对比学习）对此约束不足。

回答时要说清楚：这些是“注意力分布与错误同时出现”的观察；要证明因果，还需要做遮挡或注意力干预实验。

## 15. 梯度消失与爆炸

**面试回答：**链式法则会使多个 Jacobian 连乘，特征值/奇异值尺度长期偏小或偏大时，容易造成梯度衰减或放大。深层 Transformer 主要通过残差路径、归一化、合理初始化、学习率调度和梯度裁剪缓解。

Pre-Norm：

$$
y=x+F(\operatorname{Norm}(x)).
$$

Post-Norm：

$$
y=\operatorname{Norm}(x+F(x)).
$$

**重要纠错：**以上两个式子不能颠倒。Pre-Norm 通常使很深的网络更容易优化。

常用实务组合：梯度裁剪、warmup、稳定初始化、混合精度 loss scaling（对 FP16 尤其相关）、监控梯度范数和激活异常。

## 16. LayerNorm、BatchNorm 与 RMSNorm

**面试回答：**三者主要区别在归一化统计的计算维度和是否减均值。Transformer 序列模型通常采用 LayerNorm 或 RMSNorm，而非依赖 mini-batch 统计的 BatchNorm。

| 方法 | 归一化对象 | 是否中心化 | 是否依赖其他 batch 样本 |
| --- | --- | --- | --- |
| BatchNorm | 每个通道/特征沿 batch 等维度 | 是 | 训练时通常依赖 |
| LayerNorm | 每个 token 的隐藏特征维 | 是 | 否 |
| RMSNorm | 每个 token 的隐藏特征维 | 否 | 否 |

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,
$$

$$
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
$$

## 17. Agent 多轮对话中的 Attention 局限

**面试回答：**多轮 Agent 处理的不只是文本，还包括工具调用结果和状态变更。即使模型支持很长的上下文，长历史也可能提高推理成本、引入干扰，并导致关键约束被忽视。Attention 本身不提供可靠的持久化记忆和业务状态管理。

- **上下文限制：**窗口外历史无法直接使用。
- **信息检索：**长提示中的关键信息未必被稳定使用。
- **成本：**prefill 随输入增长；KV cache 可避免重复计算旧 token 的 K/V，但会占用显存。
- **状态一致性：**工具调用和多轮任务需要外部结构化状态与校验。

**应对：**阶段性摘要、RAG 或可检索记忆、状态机/任务数据库、工具结果压缩、关键信息置顶、输入和输出校验。

## 18. 模型并行

**面试回答：**单 GPU 无法承载模型或激活时，可从参数张量、层、序列、专家四个维度进行分片；实际大规模训练常混合这些方案与数据并行、ZeRO/FSDP。

| 策略 | 切分对象 | 主要挑战 |
| --- | --- | --- |
| Tensor Parallel (TP) | 同层权重和矩阵计算 | 高频集合通信，偏好高速互联 |
| Pipeline Parallel (PP) | 不同层/阶段 | 流水线气泡、负载平衡 |
| Sequence Parallel (SP) | 序列维度上的部分激活 | 与 TP 配合的通信切换 |
| Expert Parallel (EP) | 不同 MoE 专家 | token dispatch 与 All-to-All 通信 |
| Data Parallel (DP) | 不同数据批次的模型副本 | 梯度同步；单副本仍需容纳模型 |
| ZeRO / FSDP | 优化器、梯度和参数状态 | 分片/重建参数时的通信 |

**选择思路：**优先确认瓶颈是参数、优化器状态、激活还是算力；单层太大考虑 TP，层数太多考虑 PP，MoE 专家多考虑 EP，状态冗余考虑 FSDP/ZeRO。

---

## 19. 面试速查

| 问题 | 先答这一句 |
| --- | --- |
| Transformer 有什么？ | Attention + FFN + Residual + Norm + Position 信息 |
| Attention 公式？ | $\mathrm{softmax}(QK^\top/\sqrt{d_k}+M)V$ |
| 为什么多头？ | 多种可学习子空间并行建模相关性 |
| 为什么缩放？ | 控制点积 logit 方差，避免 Softmax 过度饱和 |
| 为什么有位置编码？ | 注意力自身不提供充分的顺序信息 |
| 为什么需要 K？ | Q/K 解耦，允许更自由的匹配函数 |
| 为什么 GPT 用 Decoder-only？ | 因果自回归目标与统一续写范式相契合 |
| 为什么 MoE？ | 用稀疏激活扩大总参数容量 |
| MoE 均衡怎么做？ | 辅助损失、专家容量或动态路由偏置 |
| Attention 复杂度？ | 典型 $O(n^2d)$；整层还含 $O(nd^2)$ |
| FlashAttention 改什么？ | 主要优化精确 attention 的 IO 与显存，不是直接线性化算术 |
| Pre-Norm 公式？ | $x+F(\mathrm{Norm}(x))$ |
| LayerNorm 与 BN 区别？ | LN 在单样本特征维归一化，BN 训练依赖 batch 统计 |
| Agent 为什么需要 Memory？ | Context attention 不等于可靠持久状态存储 |
| 主要并行方式？ | TP、PP、SP、EP、DP 和 ZeRO/FSDP |

### 建议的面试回答结构

1. **一句话定义**：说清解决什么问题。
2. **原理公式**：解释输入、操作与输出。
3. **优缺点权衡**：尤其是复杂度、显存和通信。
4. **工程落地**：说明在 LLM/Agent 场景怎样使用。
5. **边界条件**：避免“总是”“完全解决”这类绝对判断。

---

*笔记为根据原文议题撰写的学习性整理；若需逐字阅读，请访问上方原始博客。*
