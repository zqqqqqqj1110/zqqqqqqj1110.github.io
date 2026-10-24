---
title: "CityLearn介绍与初步使用"
date: 2025-10-24 23:58:26
categories:
  - CityLearn与能源
tags:
  - CityLearn
---

# CITYlearn介绍与初步使用
## 1. 介绍
CityLearn 是一个强化学习环境，用于研究 多智能体强化学习（Multi-Agent Reinforcement Learning, MARL） 在城市能源管理中的应用。
简单的来说，这是一个python的包，只不过这个包还能当模拟器用，在内部集成了建筑能耗、储能、电价、碳排放、气候等“物理模型”每执行一步 (env.step(actions))，都会模拟“1小时或15分钟的城市运行过程”，并且该环境会反馈：
· 各建筑的电力消耗；
· 储能水平；
· 奖励函数（是否节能/减碳）；
· 下一时刻的状态。

## 研究场景
Building Energy Storage Control
→ 智能体决定何时充放电以降低峰值负载。
（目标：降低电价或平滑全城功率曲线）
Multi-Agent Coordination
→ 多栋建筑通过协作学习，共同优化城市能源使用。
（目标：降低全局碳排或电网压力）
Demand Response (DR)
→ 响应动态电价或电网信号，调整能耗。
District Heating/Cooling Optimization
→ 有时扩展到热力与制冷系统的协调

## 官方推荐算法结构
```
for episode in range(num_episodes):
    observations = env.reset()
    done = False
    while not done:
        actions = [agent_i.act(obs_i) for obs_i in observations]
        next_obs, rewards, done, info = env.step(actions)
        for i, agent_i in enumerate(agents):
            agent_i.learn(observations[i], actions[i], rewards[i], next_obs[i])
        observations = next_obs
```
每个 agent_i 可以是：
· PPO agent
· DDPG agent
· 共享参数的多智能体 SAC agent
其他步骤会在AI大世界中给出，链接为：？

## 实战
### 1. conda环境
```
conda create -n citylearn python=3.10 -y

conda activate citylearn
```
官方文档里说建议 Python 3.9 或 3.10 and 不要3.12(会导致部分库（如 stable-baselines3）不兼容)

### 安装包
```
pip install citylearn

pip install numpy pandas matplotlib gym
```
当然，如果后期打算训练强化学习智能体（比如 PPO/SAC），可以提前安装SB3
```
pip install stable-baselines3
```
sb算法库，包含经典 RL 算法（如 PPO、SAC、TD3、DQN 等），可直接和 OpenAI Gym 或 CityLearn 这样的环境对接

### 数据集
官方数据集
```
git clone https://github.com/intelligent-environments-lab/CityLearn.git
cd CityLearn
```
把它down下来
### 测试
创建测试文件
```
from citylearn import CityLearnEnv
from pathlib import Path

schema_path = Path("data/citylearn_challenge_2022/schema.json")

# 创建环境
env = CityLearnEnv(schema=schema_path)

# 重置环境
observations = env.reset()
done = False

# 模拟一天
for _ in range(24):
    actions = env.action_space.sample()  # 随机动作
    observations, rewards, done, info = env.step(actions)
    print(f"Reward: {rewards:.3f}")
```
