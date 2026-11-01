---
title: "GRPO理论"
date: 2026-11-01 17:53:19
categories:
  - 面试
  - 八股
tags:
  - 强化学习
  - LLM
---

# GRPO 基础八股｜32 题标准答案（面试口述版 + 原理追问）

> 面向：大模型算法 / AI Agent 算法岗位。与对应的《GRPO 基础八股面试题清单》按题号对应。  
> 建议：先背 P0（1–15），再学 P1（16–26），P2（27–32）只需初步理解。  
> 说明：这里以 **DeepSeekMath 原始 GRPO** 为基准。实际框架可能采用 DAPO、Dr.GRPO 等改进版，某些参数（尤其 KL 和 loss 归一化）不完全一致。

## 一、统一符号和核心公式（先读这一节）

- `q`：Prompt；`o_i`：对同一 Prompt 生成的第 `i` 条回答；`G`：组内回答数量。
- `π_θ`：当前正在训练的 Policy；`π_old`：生成本轮回答的旧 Policy；`π_ref`：作为正则化基准的 Reference Policy（一般由初始/SFT 模型冻结而来）。
- `r_i`：回答 `o_i` 的标量奖励；`A_i`：回答的组相对优势。
- `ε_clip`：裁剪区间半宽；`β`：KL 惩罚系数；`δ`：标准差的数值稳定项。

**组相对优势：**

$$
\hat A_i=\frac{r_i-\bar r}{s_r+\delta},\qquad
\bar r=\frac1G\sum_{j=1}^{G}r_j,\quad
s_r=\sqrt{\frac1G\sum_{j=1}^{G}(r_j-\bar r)^2}.
$$

**Token 级新旧策略概率比：**

$$
\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\rm old}(o_{i,t}\mid q,o_{i,<t})}.
$$

**原始 GRPO 目标（最大化的形式，采用结果级 Reward）：**

$$
J_{\rm GRPO}(\theta)=\mathbb E_{q,\{o_i\}\sim\pi_{\rm old}}
\left[\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_{t=1}^{|o_i|}
\left(
\min\!\left(\rho_{i,t}\hat A_i,\operatorname{clip}(\rho_{i,t},1-\varepsilon_{\rm clip},1+\varepsilon_{\rm clip})\hat A_i\right)
-\beta\widehat D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})_{i,t}
\right)\right].
$$

训练代码通常**最小化**损失 `L = −J`。原始论文也允许按 step 计算过程奖励和相应优势；上式展示最常被问到的结果级版本。

一个常用的逐 token KL 估计为：

$$
\widehat D_{\rm KL}=\frac{\pi_{\rm ref}(o_{i,t}|h)}{\pi_\theta(o_{i,t}|h)}
-\log\frac{\pi_{\rm ref}(o_{i,t}|h)}{\pi_\theta(o_{i,t}|h)}-1,
\qquad h=(q,o_{i,<t}).
$$

---

# P0：必须掌握（1–15）

## 1. 什么是 GRPO？核心思想是什么？

**面试口述版：** GRPO 是 Group Relative Policy Optimization，即组相对策略优化，由 DeepSeekMath 提出。对于一个 Prompt，模型生成多条回答，用 Reward 评估质量，再基于同组回答的相对分数计算 Advantage。GRPO 通过鼓励组内高于平均水平的回答、抑制低于平均水平的回答来更新策略。相较 PPO，它不需要单独训练 Critic/Value Model，降低训练开销，同时继承了策略比率裁剪等稳定化思想；原始形式也含 KL 惩罚。

**追问要点：** “相对”不是只选最好一条，通常每条回答都有相应优势值；优势可能正、负或零。

## 2. GRPO 为什么提出？主要解决 PPO 的什么问题？

**面试口述版：** PPO 在大模型 RLHF 中通常训练 Policy 和 Value/Critic 两类模型，后者用来估计状态价值和 Advantage，额外消耗显存、算力，并引入价值拟合误差。GRPO 直接用同题多个回答的组内 Reward 统计量构造相对 Advantage，省去单独训练 Critic，因此在大模型训练中更节省资源。

**追问要点：** 省掉 Critic 并不代表零额外成本：GRPO 往往需要对同一个 Prompt 生成 `G` 条回答，Rollout 和奖励计算仍可能很昂贵；Reference Policy 也可能带来推理开销。

## 3. GRPO 一次完整训练流程是什么？

**面试口述版：** 首先采样一批 Prompt；利用旧策略对每个 Prompt 生成 `G` 个 Response；由规则或奖励模型打分；对每个组计算均值、标准差和相对 Advantage；再算当前策略与旧策略的 Token 概率比，通过 Clip 策略目标和可选的 Reference KL 惩罚计算损失；最后反向传播更新 Policy，并周期性重新采样 Rollout。

**顺序：** Prompt → `G` 个回答 → Reward → Group Advantage → log-prob/ratio → Clip/KL → 梯度更新。

## 4. 写出 GRPO 核心目标函数，并解释每项

**面试口述版：** GRPO 最大化组内 Token 级的 `min(ratio × advantage, clipped ratio × advantage)`，并在原始版本中减去 `β × KL(current || reference)`。其中 Ratio 衡量新旧策略对采样 Token 的概率变化；Advantage 判断这条回答相对同组好坏；Clip 限制激进更新；KL 抑制策略相对参考模型过度漂移。完整公式见文档开头。

**追问要点：** `1/G` 对组内回答求均值，`1/|o_i|` 对该回答的 Token 平均；不同后续变体会修改这一步以处理长度偏置。

## 5. Advantage 怎么计算？

**面试口述版：** 原始结果级 GRPO 对同一个 Prompt 的 `G` 条回答分别计算 Reward，取组内均值 `μ` 和标准差 `σ`，再用 `A_i=(r_i−μ)/(σ+δ)` 得到 Advantage。比均值高的回答是正优势，低的是负优势；它不需要 Value Model。

**例子：** 3 条回答 Reward 为 `[1, 0, 0]`，均值 `1/3`，总体标准差约 `0.471`，则优势约为 `[1.414, −0.707, −0.707]`。

## 6. 为什么一个 Prompt 要采样多个 Response？

**面试口述版：** 因为 GRPO 的基线来自同组回答。多条 Response 让我们能在相同问题难度背景下比较回答质量，估计组内均值和离散程度，从而判断哪些回答值得鼓励，而不是直接依赖绝对奖励值。多样性足够时，相对比较能带来有效优化信号。

**追问要点：** `G=1` 时组内方差为零，无法通过常规标准化获得有效的相对优势。

## 7. GRPO 为什么不需要 Critic？

**面试口述版：** PPO 的 Advantage 通常依赖 Value Function，例如 `A_t ≈ return_t − V(s_t)`；GRPO 把同组回答的平均 Reward 当作经验基线，通过 `Reward−group mean` 建立相对优劣，因此不必单独拟合状态价值网络。

**追问要点：** 这不是严格意义上逐状态价值函数的无偏替代；它是一种组相对优势估计，依赖同题采样的分布、Reward 质量与 Group Size。

## 8. PPO 和 GRPO 核心区别？

**面试口述版：** PPO 通常使用 Critic 估值，再利用 GAE 等方法计算时序 Advantage；GRPO 则对同一 Prompt 采样一组回答，按组内 Reward 标准化构造 Advantage，不训练额外 Critic。两者都可使用新旧策略概率比与 clipped surrogate objective。GRPO 减少 Value 模型开销，但可能增加一题多次生成的 Rollout 成本。

| 维度 | PPO（典型 LLM RLHF） | GRPO（原始结果级） |
|---|---|---|
| Advantage | Value/Critic + Return/GAE | 组内 Reward 相对标准化 |
| Critic | 通常需要 | 不需要 |
| 采样 | 可按 prompt 采样回答 | 明确使用同题 G 条回答 |
| 稳定机制 | Ratio + Clip，通常也有 KL | Ratio + Clip，原始版本有 KL |
| 代价 | Value 网络训练 | 多条回答 Rollout / 打分 |

## 9. Importance sampling ratio 是什么，为什么需要？

**面试口述版：** 新旧策略概率比 `ρ=π_θ(a|s)/π_old(a|s)` 表示当前策略相对采样策略对该动作概率的变化。Rollout 由旧策略产生，而训练时当前参数可能已经变化，因此用这个比率修正数据分布变化，并结合 Clip 防止策略更新过大。

**追问要点：** `ρ=1` 表示该 Token 的新旧条件概率相同；`ρ>1` 表示新策略更倾向该 Token。严格的轨迹级重要性采样是整条轨迹概率比；PPO/GRPO 常用逐 Token 比率形成实用的代理目标，不要称它是完全精确的轨迹分布校正。

## 10. 为什么要 Clip？不 Clip 会怎样？

**面试口述版：** 策略梯度直接乘概率比和 Advantage，个别样本可能推动新策略远离旧策略，训练容易震荡。Clip 把用于优化的概率比限制在 `[1−ε,1+ε]` 的保守代理目标内，阻止超过阈值的“有利方向”继续获得额外收益。

**追问要点：** Clip 并非硬性保证所有 Token 的 Ratio 都留在区间内，也不是严格保证 KL 的上界。

## 11. Clipped objective 为什么取 min？

**面试口述版：** 因为最大化目标需要保守估计。如果 `A>0`，模型想提高该动作概率，`ρ>1+ε` 时取裁剪后较小收益，阻止继续过度奖励；如果 `A<0`，模型想降低该动作概率，`ρ<1−ε` 时 `min` 会选更保守的裁剪项，防止继续往过低概率方向获得收益。这就是 PPO 的 pessimistic surrogate objective。

**对好回答，我们只限制概率上升过多，不限制概率下降带来的惩罚。对差回答，我们只限制概率下降过多，不限制概率上升带来的惩罚。**

**数值：** 当 `A=1, ρ=1.5, ε=0.2`，`min(1.5,1.2)=1.2`；当 `A=−1, ρ=0.5`，`min(−0.5,−0.8)=−0.8`。

## 12. KL Divergence 的作用？

**面试口述版：** KL 衡量当前 Policy 与 Reference Policy 的分布差异。原始 GRPO 用 `−β·KL` 惩罚过大的偏离，防止模型为了获得高 Reward 而过度改变语言分布、损害原有能力或出现 Reward Hacking。`β` 越大约束越强，但可能抑制学习。

**追问要点：** 有的工程版本默认不加 KL，不能说“所有 GRPO 必须有 KL”。KL 也不能单独保证模型正确或安全。

## 13. Reference、Old、Current 三种 Policy 区别？

**面试口述版：** Current Policy 是当前被梯度更新的模型 `π_θ`；Old Policy 是这批 Rollout 生成时的策略快照 `π_old`，用于计算 Ratio；Reference Policy 通常是被冻结的初始/SFT 模型 `π_ref`，用于 KL 正则。Old 会随训练批次更新，Reference 往往长期冻结，Current 则实时更新。

**记忆：** `old` 管“这一轮更新了多少”；`ref` 管“离原始模型漂移了多少”。

## 14. Reward 和 Advantage 有什么区别？为何不直接用 Reward？

**面试口述版：** Reward 是评价回答质量的原始分数，Advantage 是它相对一个基线好多少。直接把非负 Reward 作为优化权重时，可能把较差但仍得正分的回答也往上推；减掉组均值后，低于组平均的回答得到负优势，有明确的抑制信号，同时在适当条件下可以降低梯度估计方差。

**追问要点：** 组内标准差归一化也会改变不同 Prompt 的相对权重，因此“标准化一定更好”不成立。

## 15. 同组 Reward 全相同会怎样？

**面试口述版：** 组内 Reward 全相同时，`r_i−mean=0`，标准差也为零。实际实现通过稳定项或特殊处理避免除零，得到的组相对 Advantage 通常全为零，这个组不提供有效的 Policy Gradient 学习信号（如果有 KL 正则仍可能有 KL 梯度）。

**例子：** 全对 `[1,1,1,1]` 或全错 `[0,0,0,0]` 都会出现这种情况。可通过提高采样多样性、挑选难度适中的 Prompt 等手段改善。

---

# P1：高频进阶（16–26）

## 16. Group Size `G` 怎么影响训练？越大越好吗？

**面试口述版：** 较大的 `G` 提供更丰富的组内比较、通常使组均值估计更稳定，也更可能找到有差异的回答；但生成和打分成本近似随回答数增加，且收益递减。若采样结果高度同质化，增大 `G` 也未必提升训练信号，因此要在效果、探索和成本间折中。

**追问要点：** `G` 大不等于 Reward 一定更有区分度；还受采样温度与任务难度影响。

## 17. 为什么除以组内标准差，而不只减均值？

**面试口述版：** 减均值决定相对正负；除以标准差让不同组的 Advantage 尺度更接近，缓解某些组奖励范围较大的影响，便于优化。但对标准差极小的组可能放大微小差异，也可能产生题目难度偏置，因此新改进会考虑不做组内标准差缩放或改为 Batch 级缩放。

**追问要点：** 区分 `δ`（防止除零）与 PPO 的 `ε_clip`（裁剪范围）。

## 18. GRPO 是 On-policy 还是 Off-policy？

**面试口述版：** 通常把 GRPO 视作近 On-policy 的策略优化方法：使用当前/近期策略采样回答，再对这些样本进行有限次策略更新。它允许通过 `π_θ/π_old` 的比率对同一批样本做多次更新，但不适合无限重复利用很久之前的旧数据，否则策略偏移过大、比率裁剪频繁，训练信号会变差。

**追问要点：** “可以重复使用一个 Rollout Batch”不等于“任意离线数据都能直接用于标准 GRPO”。

## 19. GRPO 和 DPO 区别？

**面试口述版：** GRPO 是在线/近在线的强化学习策略优化：自己采样多条回答、打 Reward、计算组相对 Advantage，再做策略梯度；DPO 通常直接用离线的偏好对 `(chosen, rejected)`，通过相对于 Reference Policy 的对数概率差优化偏好目标，不显式训练 Reward Model，也不需要在每次优化时在线 Rollout。GRPO 更灵活地优化可验证奖励，但训练链路更复杂。

**追问要点：** DPO 可以有在线扩展版本；以上对比针对经典 DPO。

## 20. GRPO 和 SFT 的目标区别？

**面试口述版：** SFT 是监督学习，通过最小化示范答案 Token 的负对数似然，学习“示范者怎么回答”；GRPO 则从模型自己生成的候选回答中，按 Reward 相对优劣提高或降低生成概率，优化“什么样的回答更值得产生”。SFT 依赖高质量示范，GRPO 依赖有效奖励和探索。

**记忆：** `SFT = imitation`，`GRPO = reward-driven policy optimization`。

## 21. 为什么 GRPO 常用于数学、代码和推理？

**面试口述版：** 数学和代码常能构造较客观的 Outcome Reward，例如答案是否正确、单元测试是否通过。对同一题生成多个推理路径后，GRPO 可以通过组内相对质量强化更有效的推理方式，且不必单独训练 Critic。其效果仍受验证器准确度、任务难度和采样多样性限制。

## 22. Sequence-level Reward 怎么作用到 Token？

**面试口述版：** 结果级 GRPO 先对整个 Response 给一个 `r_i`，算出序列 Advantage `A_i`，然后在该回答的每个生成 Token 上使用同一个 `A_i`，对每个 Token 的新旧策略概率比计算 clipped policy objective，再按序列长度平均。这样整条高分回答的生成行为获得正向训练信号。

**追问要点：** 结果级归因比较粗糙：正确答案中的冗余/错误中间步骤也可能分享正 Advantage。这是过程奖励和更精细 Credit Assignment 的动机之一。

## 23. KL 和 Clip 都限更新，有何区别？

**面试口述版：** Clip 限制的是训练样本上 `current / old` 比率的代理收益，重点约束当前一轮相对 Rollout Policy 的更新幅度；KL 通常比较 `current` 与固定的 `reference`，约束长期偏离。Clip 通过目标函数的分段截断起作用，KL 是额外的正则惩罚，两者可互补，并不能互相完全替代。

## 24. GRPO 不稳定有哪些原因？

**面试口述版：** 主要检查：奖励是否稀疏/噪声大、同组是否全同分、Reward Scale 和标准化方式、Learning Rate 是否过大、Clip Fraction 是否异常、KL 是否过大或过小、Rollout 长度和多样性是否变化、训练/采样 Policy 是否错位。我的排查顺序通常是先核对 Reward 和 Advantage 分布，再看 Ratio/Clip/KL，最后调整超参数。

## 25. Reward Hacking 是什么？为什么发生？

**面试口述版：** Reward Hacking 是模型学到“拿高奖励的捷径”，而不是完成真实目标。例如验证规则有漏洞，模型就反复输出某些能误导评分器的模式。GRPO 直接优化 Reward，因此同样可能不断放大奖励函数的缺陷。应通过更可靠的验证器、多个奖励维度、未见任务评测与人工抽检缓解。

## 26. 怎么判断 GRPO 训练是否有效？

**面试口述版：** 不能只看训练 Reward 上升，还要检查独立测试集的 Pass Rate/Accuracy、输出格式正确率、Reward 各分项、回答长度、样本多样性，以及 KL、Clip Fraction、Advantage 方差、零方差组比例等指标。只有独立任务质量提升、没有明显能力退化，才更能说明训练有效。

**追问要点：** 比较不同 Checkpoint 时应固定评测集和解码配置，避免采样随机性造成假提升。

---

# P2：了解即可（27–32）

## 27. GRPO 和 DAPO 有什么区别？

**面试口述版：** DAPO 可以理解为面向长链推理训练对原始 GRPO 的一套改进，代表性措施包括更灵活的裁剪、动态采样、Token 级 Loss 归一化和过长样本处理，目标是提升训练稳定性和有效 Token 利用率。面试先把原始 GRPO 讲清楚，再区分这些属于后续改进，不要说它们是 GRPO 原始公式自带的。

## 28. GRPO 和 GSPO 区别？

**面试口述版：** 经典 GRPO 通常基于 Token 级概率比和裁剪目标做优化；GSPO（Group Sequence Policy Optimization）把重要性权重/裁剪更多放到序列级处理，旨在使整条回答的优化更一致，改善部分训练场景下的稳定性。具体数学定义依论文实现，不应把 Token 比率与序列比率简单当成完全等价。

## 29. Token 级 Clip 与 Sequence 级优化的争议？

**面试口述版：** 当一个序列只有结果级奖励时，每个 Token 共享同一个优势，但不同 Token 的 Ratio 和 Clip 状态可能不同，产生较复杂的信用分配与长度效应。序列级方法希望让整条 Response 的优化权重更一致，但可能更难处理高方差的整轨迹概率比，所以需要做稳定性折中。

## 30. Agentic RL 的过程奖励怎么设计？

**面试口述版：** 除最终成功率外，还可以针对工具调用有效性、参数合法性、执行结果核验、约束遵守、步骤成本等提供过程反馈。关键是有客观证据，避免奖励“思考步骤多”这种易被刷分的代理指标，并用独立任务成功率验证。

## 31. 长 Rollout 影响吞吐，如何优化？

**面试口述版：** 可从采样引擎加速、批处理与连续批次推理、合理的最大输出长度、异步 Rollout、缓存与资源调度、动态采样剔除无信息组等方向改善。但所有优化都要核查训练策略与采样策略的版本差异，以及数据分布偏移。

## 32. 手写简化版 GRPO Loss

**面试口述版：** 先对同 Prompt 的奖励做组内标准化得到 `A`；计算当前和旧策略在已采样 Token 上的 `log_prob`，两者相减再取指数得到 `ratio`；求 `minimum(ratio*A, clip(ratio)*A)`；只对有效生成 Token 求均值；若启用参考模型约束，再加上 KL 项；最后取负号做梯度下降。

**伪代码（示意，非可直接生产训练的完整实现）：**

```python
# rewards: [B, G]
# logp_new, logp_old, logp_ref: [B, G, T]
# mask: [B, G, T] (completion tokens only)
mu = rewards.mean(dim=1, keepdim=True)
std = rewards.std(dim=1, correction=0, keepdim=True)
adv = (rewards - mu) / (std + 1e-6)
adv = adv.unsqueeze(-1)            # [B,G,1], broadcast to tokens
ratio = torch.exp(logp_new - logp_old)
clipped = torch.clamp(ratio, 1-eps_clip, 1+eps_clip)
surrogate = torch.minimum(ratio * adv, clipped * adv)
log_r = logp_ref - logp_new
kl = torch.exp(log_r) - log_r - 1
per_response = ((-surrogate + beta * kl) * mask).sum(-1) / mask.sum(-1).clamp_min(1)
loss = per_response.mean()
```

**注意：** 原始目标有多种改进型归一化；实训需处理 Padding、EOS、长度截断、过时 Rollout、Reward 标准化和数值稳定等问题。

---

# 复习路线与易错点

**最短背诵链（面试前 10 分钟）：** Q1 → Q3 → Q5 → Q7 → Q9 → Q10 → Q11 → Q12 → Q13 → Q8 → Q19。

**五个常见错误：**

1. “GRPO 没有 Reward Model”——不严谨。它**不需要 Critic**，Reward 可以来自规则、验证器或奖励模型。
2. “GRPO 的 Reference 和 Old 是同一个模型”——不严谨。前者一般固定用于 KL，后者是采样时策略快照。
3. “GRPO 的 KL 一定不可以为零”——不对，部分实现允许 `β=0`。
4. “Clip 就能保证策略永远不会走远”——不对，Clip 是代理目标，不是严格的全局约束。
5. “组内标准差越小越能学到东西”——不对，全同分时无组相对策略梯度；极小标准差还可能放大噪声。

# 参考资料（原理优先于二次转述）

- DeepSeekMath 原论文：https://arxiv.org/abs/2402.03300
- Hugging Face TRL / GRPO Trainer：https://huggingface.co/docs/trl/grpo_trainer
- 原始 GRPO 公式解读：https://learn.arena.education/chapter2_rl/04_rlhf/1-rlhf-via-grpo/

> 文档定位为面试学习资料，不是对某公司面试题库或某一工程实现的逐字复刻。
