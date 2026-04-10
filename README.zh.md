# SocialMeta

**面向序贯社会困境的快速 GPU 加速多智能体强化学习与元学习框架**

[English](README.md) | [中文](README.zh.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.23+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SocialMeta 是一个基于 [JAX](https://github.com/google/jax) 构建的高性能基准测试套件，用于在序贯社会困境（Sequential Social Dilemmas, SSDs）中评估多智能体强化学习（MARL）和元强化学习（Meta-RL）算法。本项目基于优秀的 [SocialJax](https://github.com/cooperativex/SocialJax) 代码库构建，扩展了完整的元学习能力以及增强的评估工具。

## 🚀 核心特性

- **⚡ 纯 JAX 实现**: 端到端 JIT 编译，最大化 GPU/TPU 性能
- **🎯 元学习支持**: 内置 RL² 和 MAML 元学习算法
- **🤝 多样化的 SSD 环境**: 8 个具有挑战性的多智能体社会困境环境
- **📊 全面的评估工具**: 评估策略对不同队友的泛化能力
- **🔧 统一接口**: 单一命令即可在任何环境上训练任意算法
- **🔌 Hydra 配置管理**: 灵活、可组合的实验配置
- **📈 W&B 集成**: 内置实验追踪与可视化

## 📋 环境要求

- Python >= 3.10
- JAX >= 0.4.23（需要 CUDA 支持以进行 GPU 训练）
- 完整依赖见 `requirements.txt`

## 🔧 安装

### 使用 Conda（推荐）

```bash
# 创建并激活环境
conda create -n social-meta python=3.10
conda activate social-meta

# 克隆仓库
git clone <repository-url>
cd socialmeta

# 安装依赖
pip install -r requirements.txt
```

### 使用 Poetry

```bash
# 安装依赖
poetry install

# 激活环境
poetry shell
```

### 验证安装

```bash
# 测试 JAX GPU 支持
python -c "import jax; print(jax.devices())"

# 测试环境加载
python -c "import socialmeta; env = socialmeta.make('coop_mining'); print('环境加载成功!')"
```

## 🎮 环境介绍

SocialMeta 包含 8 个序贯社会困境环境：

| 环境 | 描述 | 智能体数 | 核心挑战 |
|------|------|---------|---------|
| **Cleanup（清洁）** | 清理污染同时避免过度开采 | 7 | 公共品困境 |
| **Coin Game（硬币游戏）** | 收集具有不对称激励的硬币 | 2 | 利益冲突 |
| **Common Harvest（共同收割）** | 可持续资源管理 | 4-7 | 公地悲剧 |
| **Coop Mining（合作采矿）** | 协作开采铁矿和金矿 | 6 | 不确定性下的合作 |
| **Gift（礼物交换）** | 交换礼物以最大化福利 | 2-4 | 互惠行为 |
| **Mushrooms（蘑菇）** | 采集有毒风险的食物 | 2 | 风险评估 |
| **PD Arena（囚徒困境竞技场）** | 空间迭代囚徒困境 | 2-8 | 合作涌现 |
| **Territory（领地争夺）** | 占领和防守领地 | 2-4 | 竞争与协调 |

所有环境支持：
- CNN 观测处理（基于网格）
- 个体或共享奖励
- 可配置的回合长度
- JIT 编译的 step 函数

## 🏃 快速开始

### 统一训练接口

SocialMeta 提供了统一的命令行接口来训练所有算法：

```bash
# 列出所有支持的算法
python train.py --list_algos

# 列出某算法支持的环境
python train.py --algo IPPO --list_envs
```

### 训练示例

#### IPPO（独立 PPO）

```bash
# 快速测试运行
python train.py --algo IPPO --env coop_mining --test

# 完整训练
python train.py --algo IPPO --env coop_mining \
    --total_timesteps 3e8 \
    --num_envs 256 \
    --lr 0.0005

# 在不同环境上训练
python train.py --algo IPPO --env cleanup --test
python train.py --algo IPPO --env coins --test
```

#### RL²（带记忆的元学习）

首先，下载或训练 SVO 队友策略：

```bash
bash get_svo_policies.sh
```

然后训练 RL²：

```bash
# 快速测试
python train.py --algo RL2 --env coop_mining --test

# 完整训练
python train.py --algo RL2 --env coop_mining \
    --total_timesteps 3e8 \
    --num_envs 512 \
    --trial_episodes 3 \
    --episode_reward_weights "[0.2,0.3,0.5]"
```

#### MAML（模型无关元学习）

```bash
# 快速测试（推荐使用一阶近似以降低内存）
python train.py --algo MAML --env coop_mining --test --first_order_maml

# 完整训练
python train.py --algo MAML --env coop_mining \
    --total_timesteps 3e8 \
    --num_envs 512 \
    --lr 1e-4 \
    --first_order_maml
```

#### 其他算法

```bash
# MAPPO
python train.py --algo MAPPO --env coop_mining --test

# 不带 SVO 包装器的 IPPO
python train.py --algo IPPO_raw --env coop_mining --test
```

### 通用参数

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--algo` | 算法：IPPO, IPPO_raw, MAPPO, RL2, MAML, SVO | 必需 |
| `--env` | 环境名称 | 必需 |
| `--total_timesteps` | 总训练步数 | 50000 |
| `--num_envs` | 并行环境数 | 32 |
| `--num_steps` | 每次更新的步数 | 384-1000 |
| `--lr` | 学习率 | 0.0003-0.0005 |
| `--seed` | 随机种子 | 30 |
| `--wandb_mode` | W&B 日志：online/offline/disabled | disabled |
| `--tune` | 启用超参数搜索 | False |
| `--test` | 快速测试模式（1000 步） | False |

### 元学习专用参数

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--trial_episodes` | 每个 trial 的回合数 | 3 |
| `--episode_reward_weights` | 每回合的奖励权重 | [0.2,0.3,0.5] |
| `--first_order_maml` | 使用一阶 MAML | False |

### 使用超参数搜索进行训练

启用 wandb 超参数搜索：

```bash
python train.py --algo IPPO --env coop_mining --tune
```

## 📊 评估

评估训练好的策略对不同队友策略的泛化能力：

```bash
cd evaluation/coop_mining

# 评估训练好的 checkpoint
python evaluate.py \
    --checkpoint /path/to/checkpoint.pkl \
    --num_episodes 100
```

评估支持：
- 与 SVO 训练策略的交叉对局
- 对未见过的队友的泛化测试
- 不同队友配置下的性能指标

## 🏗️ 项目结构

```
socialmeta/
├── train.py                    # 统一训练接口 ⭐
├── socialmeta/                 # 核心库
│   ├── environments/           # SSD 环境实现
│   │   ├── cleanup/            # 清洁环境
│   │   ├── coin_game/          # 硬币游戏
│   │   ├── common_harvest/     # 共同收割
│   │   ├── coop_mining/        # 合作采矿
│   │   ├── gift/               # 礼物交换
│   │   ├── mushrooms/          # 蘑菇环境
│   │   ├── pd_arena/           # 囚徒困境竞技场
│   │   └── territory/          # 领地争夺
│   ├── wrappers/               # 环境包装器
│   └── registration.py         # 环境注册表
│
├── algorithms/                 # MARL 和 Meta-RL 实现
│   ├── IPPO/                   # 独立 PPO 基线（10 个环境）
│   ├── IPPO_raw/               # 无 SVO 的 IPPO（10 个环境）
│   ├── MAPPO/                  # 多智能体 PPO（10 个环境）
│   ├── MAML/                   # 模型无关元学习（1 个环境）
│   ├── PPO/                    # 元学习 PPO 变体（1 个环境）
│   ├── RL2/                    # RL² 循环元学习（1 个环境）
│   └── SVO/                    # 社会价值取向策略（10 个环境）
│
├── evaluation/                 # 评估脚本
│   ├── cleanup/
│   ├── coin/
│   ├── coop_mining/
│   ├── gift/
│   ├── harvest_closed/
│   ├── harvest_partnership/
│   ├── mushroom/
│   └── pd_arena/
│
├── checkpoints/                # 保存的模型检查点
├── fixed_policy/               # 脚本化基线策略
└── speed_test/                 # 性能基准测试
```

## ⚙️ 配置

所有算法使用 [Hydra](https://hydra.cc/) 进行配置管理。配置文件存储在 `algorithms/<算法名>/config/`。

### 算法-环境兼容性

| 算法 | 支持的环境 | 数量 |
|-----------|------------------------|-------|
| IPPO | coop_mining, cleanup, coins, gift, mushrooms, pd_arena, harvest_common, harvest_common_closed, harvest_common_partnership, territory_open | 10 |
| IPPO_raw | 同 IPPO | 10 |
| MAPPO | coop_mining, cleanup, coins, gifts, mushrooms, pd_arena, harvest_common, harvest_common_closed, harvest_common_partnership, territory_open | 10 |
| RL² | coop_mining | 1 |
| MAML | coop_mining | 1 |
| SVO | coop_mining, cleanup, coin, gift, mushroom, pd_arena, harvest_open, harvest_closed, harvest_partnership, territory_open | 10 |

### 通过 Hydra 进行高级配置

对于高级用例，您仍可直接调用算法脚本：

```bash
cd algorithms/IPPO
python ippo_cnn_coop_mining.py \
    LR=0.001 \
    NUM_ENVS=512 \
    ENV_KWARGS.num_agents=8 \
    ENV_KWARGS.shared_rewards=True \
    WANDB_MODE=online \
    TUNE=False
```

### 环境配置

每个环境接受 `ENV_KWARGS` 参数：

```yaml
ENV_KWARGS:
  num_agents: 6              # 智能体数量
  num_inner_steps: 1000      # 每回合步数
  num_outer_steps: 3         # 每 trial 的回合数（元学习）
  shared_rewards: false      # 个体或共享奖励
  cnn: true                  # 使用 CNN 观测
  jit: true                  # JIT 编译环境
```

### PPO 超参数（所有算法通用）

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `TOTAL_TIMESTEPS` | 总训练步数 | 3e8 |
| `NUM_ENVS` | 并行环境数 | 256-512 |
| `NUM_STEPS` | 每次更新的步数 | 384-1000 |
| `LR` / `OUTER_LR` | 学习率 | 0.0003-0.0005 |
| `GAMMA` | 折扣因子 | 0.99 |
| `GAE_LAMBDA` | GAE lambda | 0.95 |
| `CLIP_EPS` | PPO 裁剪 epsilon | 0.2 |
| `ENT_COEF` | 熵系数 | 0.01 |
| `VF_COEF` | 价值函数系数 | 0.5 |
| `MAX_GRAD_NORM` | 梯度裁剪 | 0.5 |
| `ACTIVATION` | 激活函数 | "relu" |
| `ANNEAL_LR` | 学习率退火 | True |
| `SEED` | 随机种子 | 30 |
| `TUNE` | 启用超参数搜索 | False |
| `WANDB_MODE` | W&B 日志模式 | "online" |

## 📈 实验追踪

SocialMeta 集成 [Weights & Biases](https://wandb.ai/) 进行实验追踪：

```bash
# 登录 W&B
wandb login

# 使用 W&B 日志进行训练
python train.py --algo IPPO --env coop_mining \
    --wandb_mode online
```

记录的指标包括：
- 回合回报（个体和团队）
- 策略熵
- 价值函数损失
- 评估性能（元学习）
- 训练吞吐（步数/秒）

## 🧪 可复现性

设置随机种子以获得可复现的实验：

```bash
# 单一种子
python train.py --algo IPPO --env coop_mining --seed 42

# 多种子（顺序运行）
for seed in 40 41 42 43 44; do
    python train.py --algo IPPO --env coop_mining --seed $seed
done
```

## 💡 性能优化建议

### GPU 利用率

- 增加 `--num_envs` 以提高 GPU 利用率（A100 推荐 512-1024）
- 使用 `ENV_SCAN_UNROLL` 和 `RNN_SCAN_UNROLL` 调整编译与内存的权衡

### 元学习优化

- MAML 从 `--first_order_maml` 开始（内存使用大大降低）
- 确保 `TEAMMATE_POLICY_DIR` 中的队友策略具有多样性
- 使用 `EVAL_DURING_TRAIN=True` 在训练期间监控泛化能力

### 内存优化

大规模实验：

```bash
python train.py --algo MAML --env coop_mining \
    --first_order_maml
```

或直接调用脚本：

```bash
cd algorithms/MAML
python maml_cnn_coop_mining.py \
    MAML_LOSS_REMAT=true \
    MAML_NUM_MINIBATCHES=4 \
    ENV_SCAN_UNROLL=2 \
    GAE_SCAN_UNROLL=8
```

### 按硬件推荐的配置

**8GB GPU（如 RTX 3070）：**
```bash
python train.py --algo IPPO --env coop_mining --num_envs 128 --num_steps 512
python train.py --algo MAML --env coop_mining --num_envs 64 --first_order_maml
```

**24GB GPU（如 RTX 3090）：**
```bash
python train.py --algo IPPO --env coop_mining --num_envs 512 --num_steps 1000
python train.py --algo RL2 --env coop_mining --num_envs 512
```

**40GB+ GPU（如 A100）：**
```bash
python train.py --algo IPPO --env coop_mining --num_envs 1024 --num_steps 1000
python train.py --algo MAML --env coop_mining --num_envs 512
```

## 🐛 已知问题

| 问题 | 影响 | 解决方案 |
|-------|----------|------------|
| SVO shape 错误 | SVO 算法 | 使用 IPPO_raw 替代 |
| Territory 属性错误 | Territory 环境 | 使用其他环境 |
| MAPPO GIF 保存错误 | MAPPO 评估 | 训练正常，GIF 是可选的 |
## 📚 引用

如果您在研究中使用了 SocialMeta，请同时引用本工作和原始 SocialJax：

```bibtex
@software{socialmeta2025,
  title = {SocialMeta: Meta-Reinforcement Learning for Sequential Social Dilemmas},
  url = {https://github.com/your-repo/socialmeta},
  year = {2025}
}

@software{socialjax2024,
  title = {SocialJax: JAX-based Multi-Agent RL for Social Dilemmas},
  author = {Social AI Lab},
  url = {https://github.com/cooperativex/SocialJax},
  year = {2024}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

SocialMeta 基于优秀的 [SocialJax](https://github.com/cooperativex/SocialJax) 代码库构建。我们感谢原作者为多智能体社会困境强化学习研究提供的高质量基础。

额外灵感来源：
- [PureJaxRL](https://github.com/luchris429/purejaxrl) 提供 JAX 原生 RL 实现
- [JaxMARL](https://github.com/FLAIROx/JaxMARL) 提供多智能体环境设计
