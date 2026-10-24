---
title: "强化学习7 DQN进阶（DDQN）"
date: 2026-03-09 18:00:00
categories:
  - 强化学习
---

# DQN进阶

## 1. 双深度Q网络（double DQN，DDQN）

为了解决DQN中高估（overestimation）的问题，引入了DDQN，标准DQN中

$$
y = r +\gamma \max_{a'}Q_{target}(s',a')
$$

**DQN中的Q值为**

$$
Q_{target}(s', \arg \max_{a'}Q_{target}(s', a'))
$$

**DDQN中的Q值为**

$$
Q_{target}(s', \arg \max_{a'}Q_{main}(s', a'))
$$

意思就是main网络选择了动作A，然后让target网络计算A多少分。主要用到的思想就是**交叉验证**

## 1.2 竞争深度Q网络（dueling DQN）

## 1.3 优先级经验回放（prioritized experience replay，PER）

## 1.4 彩虹（rainbow）

![image-20251230140758802](/images/rl/image-20251230140758802.png)

我觉得这是一个好方法，尤其可以用在消融实验里，所有的办法都用的就是彩虹，后续可以去掉一个模块，进行分门对比
