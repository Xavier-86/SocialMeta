# SocialMeta

> **GPU 加速的多智能体强化学习框架，面向序贯社会困境**

[![Python](https://img.shields.io/badge/Python-3.10+-2563EB?style=flat&logo=python&logoColor=white)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-0.4.23+-FB8C00?style=flat)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat)](LICENSE)

[English](README.md) · [中文](README.zh.md) · [CLI 指南](CLI_GUIDE.zh.md)

---

## 概述

SocialMeta 是一个高性能研究框架，用于多智能体强化学习（MARL）和元强化学习，专注于序贯社会困境（Sequential Social Dilemmas, SSDs）。

基于 **JAX** 构建，端到端 JIT 编译，最大化 GPU/TPU 性能。

---

## 快速开始

```bash
# 安装
conda create -n social-meta python=3.10
conda activate social-meta
pip install -r requirements.txt

# 训练 IPPO
python train.py --algo IPPO --env coop_mining --test

# 训练元学习（RL²）
python train.py --algo RL2 --env coop_mining --test
```

---

## 配置令牌 — 参数参考

### 环境矩阵

| 环境 | 智能体数 | 核心机制 | 适用场景 |
|------|---------|----------|----------|
| **Cleanup** | 7 | 公共品 + 污染 | 压力下的合作 |
| **Coin Game** | 2 | 不对称激励 | 冲突解决 |
| **Common Harvest** | 4-7 | 资源枯竭 | 可持续发展 |
| **Coop Mining** | 6 | 互补技能 | 技能专化 |
| **Gift** | 2-4 | 互惠交换 | 信任建立 |
| **Mushrooms** | 2 | 风险 + 不确定性 | 安全探索 |
| **PD Arena** | 4 | 空间迭代囚徒困境 | 规范涌现 |

### 算法能力矩阵

| 算法 | 类型 | 支持环境 | 元学习 |
|------|------|----------|--------|
| **IPPO** | MARL | 10 | ❌ |
| **IPPO_raw** | MARL | 10 | ❌ |
| **MAPPO** | MARL | 10 | ❌ |
| **SVO** | 策略库 | 10 | ❌ |
| **RL2** | 循环网络 | 6 | ✅ 基于 Trial |
| **MAML** | 梯度元学习 | 6 | ✅ 基于梯度 |

---

## 使用模式

### 模式 A: 标准 MARL 训练

```bash
python train.py \
    --algo IPPO \
    --env coop_mining \
    --num_envs 512 \
    --total_timesteps 3e8
```

### 模式 B: 元学习训练

```bash
# 步骤 1: 准备队友策略
bash get_svo_policies.sh

# 步骤 2: 使用元学习训练
python train.py \
    --algo RL2 \
    --env coop_mining \
    --trial_episodes 3 \
    --episode_reward_weights "[0.2,0.3,0.5]"
```

### 模式 C: 超参数搜索

```bash
python train.py \
    --algo IPPO \
    --env coop_mining \
    --tune
```

---

## 配置令牌

### 硬件优化预设

| GPU 显存 | NUM_ENVS | NUM_STEPS | 适用算法 |
|----------|----------|-----------|----------|
| 8 GB | 128 | 512 | IPPO |
| 8 GB | 64 | 384 | MAML (一阶) |
| 24 GB | 512 | 1000 | IPPO |
| 24 GB | 512 | 384 | RL2 |
| 40 GB+ | 1024 | 1000 | 任意 |

### 核心训练参数

```yaml
# PPO 基础
TOTAL_TIMESTEPS: 3e8        # 总训练步数
NUM_ENVS: 512               # 并行环境数
NUM_STEPS: 384              # 每次更新步数
LR: 0.0003                  # 学习率
GAMMA: 0.99                 # 折扣因子
GAE_LAMBDA: 0.95            # GAE 参数
CLIP_EPS: 0.2               # PPO 裁剪范围
ENT_COEF: 0.01              # 熵奖励系数
VF_COEF: 0.5                # 价值损失权重

# 元学习
TRIAL_EPISODES: 3           # 每 trial 的回合数
EPISODE_REWARD_WEIGHTS:     # 每回合的奖励权重
  - 0.2
  - 0.3
  - 0.5
```

### 环境配置

```yaml
ENV_KWARGS:
  num_agents: 6             # 智能体数量
  num_inner_steps: 1000     # 每回合步数
  num_outer_steps: 3        # 每 trial 的回合数
  shared_rewards: false     # 奖励结构
  cnn: true                 # 观测类型
  jit: true                 # JIT 编译
```

---

## 项目结构

```
socialmeta/
├── train.py                 # ⭐ 统一训练接口
├── socialmeta/              # 核心库
│   ├── environments/        # 8 个 SSD 环境
│   ├── wrappers/            # 观测包装器
│   └── registration.py      # 环境注册表
├── algorithms/              # MARL 实现
│   ├── IPPO/                # 独立 PPO (10 环境)
│   ├── MAPPO/               # 多智能体 PPO (10 环境)
│   ├── RL2/                 # RL² 元学习 (6 环境)
│   ├── MAML/                # MAML 元学习 (6 环境)
│   └── SVO/                 # SVO 策略库 (10 环境)
├── evaluation/              # 交叉评估
├── checkpoints/             # 模型检查点
└── svo-policies/            # 队友策略库
```

---

## CLI 参考

```bash
# 状态概览
./socialmeta-cli status

# 使用指定资源训练
./socialmeta-cli train \
    --algo IPPO \
    --env coop_mining \
    --num-envs 512

# 列出可用资源
./socialmeta-cli list
./socialmeta-cli list --svo
./socialmeta-cli list --checkpoints
```

详见 [CLI_GUIDE.zh.md](CLI_GUIDE.zh.md)

---

## 引用

```bibtex
@software{socialmeta2025,
  title = {SocialMeta: SSDs 的元强化学习框架},
  year = {2025},
  url = {https://github.com/your-repo/socialmeta}
}

@software{socialjax2024,
  title = {SocialJax: 基于 JAX 的多智能体 RL},
  author = {Social AI Lab},
  year = {2024},
  url = {https://github.com/cooperativex/SocialJax}
}
```

---

## 许可证

MIT 许可证 — 详见 [LICENSE](LICENSE)
