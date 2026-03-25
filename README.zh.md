# SocialMeta

**面向序贯社会困境的快速 GPU 加速多智能体强化学习与元学习框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.23+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SocialMeta 是一个基于 [JAX](https://github.com/google/jax) 构建的高性能基准测试套件，用于在序贯社会困境（Sequential Social Dilemmas, SSDs）中评估多智能体强化学习（MARL）和元强化学习（Meta-RL）算法。本项目基于优秀的 [SocialJax](https://github.com/cooperativex/SocialJax) 代码库构建，扩展了完整的元学习能力以及增强的评估工具。

## 🚀 核心特性

- **⚡ 纯 JAX 实现**: 端到端 JIT 编译，最大化 GPU/TPU 性能
- **🎯 元学习支持**: 内置 RL² 和 MAML 元学习算法
- **🤝 多样化的 SSD 环境**: 8 个具有挑战性的多智能体社会困境环境
- **📊 全面的评估工具**: 评估策略对不同队友的泛化能力
- **🔧 Hydra 配置管理**: 灵活、可组合的实验配置
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

### 环境设置

所有算法需要将项目根目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/socialmeta:$PYTHONPATH
```

或在算法目录内正确设置路径后运行。

### 训练 IPPO（独立 PPO）

IPPO 是标准的多智能体强化学习基线，每个智能体独立使用 PPO 学习。

```bash
cd algorithms/IPPO

# 快速测试运行
python ippo_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=50000 \
    NUM_ENVS=32 \
    WANDB_MODE=disabled \
    TUNE=False

# 完整训练并启用 wandb 日志
python ippo_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=3e8 \
    NUM_ENVS=256 \
    LR=0.0005 \
    TUNE=False

# 训练不同环境
python ippo_cnn_cleanup.py TOTAL_TIMESTEPS=3e8 TUNE=False
python ippo_cnn_coins.py TOTAL_TIMESTEPS=3e8 TUNE=False
```

**关键参数说明：**
- `TOTAL_TIMESTEPS`：总训练步数
- `NUM_ENVS`：并行环境数（越高 GPU 利用率越好）
- `NUM_STEPS`：每次更新的步数（默认：1000）
- `LR`：学习率（默认：0.0005）
- `REWARD`：`"individual"`（个体）或 `"common"`（共享）奖励结构
- `TUNE`：设为 `False` 进行单轮运行，`True` 启用超参数搜索
- `WANDB_MODE`：`"online"`（在线）、`"offline"`（离线）或 `"disabled"`（禁用）

### 训练 RL²（带记忆的元学习）

RL² 使智能体能够利用循环记忆在部署期间适应新队友。

首先，下载或训练 SVO 队友策略：

```bash
cd /path/to/socialmeta
bash get_svo_policies.sh
```

然后训练 RL²：

```bash
cd algorithms/RL2

# 快速测试
python rl2_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=50000 \
    NUM_ENVS=32 \
    TRIAL_EPISODES=3 \
    EPISODE_REWARD_WEIGHTS=[0.2,0.3,0.5] \
    WANDB_MODE=disabled \
    TUNE=False

# 完整训练
python rl2_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=3e8 \
    NUM_ENVS=512 \
    NUM_STEPS=384 \
    TRIAL_EPISODES=3 \
    EPISODE_REWARD_WEIGHTS=[0.2,0.3,0.5] \
    RNN_HIDDEN_SIZE=128 \
    TUNE=False
```

**元学习专用参数：**
- `TRIAL_EPISODES`：每个 trial（元回合）的回合数
- `EPISODE_REWARD_WEIGHTS`：trial 中各回合的权重（列表长度必须与 `TRIAL_EPISODES` 匹配）
- `RNN_HIDDEN_SIZE`：LSTM 隐藏层维度（默认：128）
- `TEAMMATE_POLICY_DIR`：包含队友策略的目录
- `EVAL_DURING_TRAIN`：在训练期间启用定期评估
- `EVAL_TIMES`：训练期间的评估检查点次数

### 训练 MAML（模型无关元学习）

MAML 学习一个良好的初始化，可以通过梯度更新快速适应新队友。

```bash
cd algorithms/MAML

# 快速测试（推荐使用一阶近似以降低内存）
python maml_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=50000 \
    NUM_ENVS=32 \
    TRIAL_EPISODES=3 \
    EPISODE_REWARD_WEIGHTS=[0.2,0.3,0.5] \
    FIRST_ORDER_MAML=true \
    WANDB_MODE=disabled \
    TUNE=False

# 完整训练
python maml_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=3e8 \
    NUM_ENVS=512 \
    OUTER_LR=1e-4 \
    INNER_LR=2e-3 \
    INNER_STEPS=1 \
    FIRST_ORDER_MAML=true \
    TUNE=False
```

**MAML 专用参数：**
- `OUTER_LR`：外循环的元学习率
- `INNER_LR`：内循环适应的学习率
- `INNER_STEPS`：内循环的梯度步数
- `FIRST_ORDER_MAML`：使用一阶近似（推荐，节省内存）
- `MAML_NUM_MINIBATCHES`：梯度累积分片数以提高内存效率
- `MAML_LOSS_REMAT`：使用梯度检查点以降低内存使用

### 使用超参数搜索进行训练

设置 `TUNE=True` 以启用 wandb 超参数搜索：

```bash
python ippo_cnn_coop_mining.py TUNE=True
```

这将运行多个实验，使用配置中定义的不同学习率和其他超参数。

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
│   ├── IPPO/                   # 独立 PPO 基线
│   ├── IPPO_raw/               # 无 SVO 的 IPPO
│   ├── MAPPO/                  # 多智能体 PPO
│   ├── MAML/                   # 模型无关元学习
│   ├── PPO/                    # 元学习 PPO 变体
│   ├── RL2/                    # RL² 循环元学习
│   └── SVO/                    # 社会价值取向策略
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

### 示例：自定义训练

```bash
# 修改多个参数
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

### 所有算法的通用参数

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
python ippo_cnn_coop_mining.py \
    ENTITY=your-username \
    PROJECT=socialmeta-experiments \
    WANDB_TAGS=["baseline","coop_mining"] \
    TUNE=False
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
python ippo_cnn_coop_mining.py SEED=42 TUNE=False

# 多种子（顺序运行）
for seed in 40 41 42 43 44; do
    python ippo_cnn_coop_mining.py SEED=$seed TUNE=False
done
```

## 💡 性能优化建议

### GPU 利用率

- 增加 `NUM_ENVS` 以提高 GPU 利用率（A100 推荐 512-1024）
- 使用 `ENV_SCAN_UNROLL` 和 `RNN_SCAN_UNROLL` 调整编译与内存的权衡

### 元学习优化

- MAML 从 `FIRST_ORDER_MAML=True` 开始（内存使用大大降低）
- 确保 `TEAMMATE_POLICY_DIR` 中的队友策略具有多样性
- 使用 `EVAL_DURING_TRAIN=True` 在训练期间监控泛化能力

### 内存优化

大规模实验：

```bash
python maml_cnn_coop_mining.py \
    MAML_LOSS_REMAT=true \
    MAML_NUM_MINIBATCHES=4 \
    ENV_SCAN_UNROLL=2 \
    GAE_SCAN_UNROLL=8
```

### 按硬件推荐的配置

**8GB GPU（如 RTX 3070）：**
```bash
python ippo_cnn_coop_mining.py NUM_ENVS=128 NUM_STEPS=512 TUNE=False
python maml_cnn_coop_mining.py NUM_ENVS=64 FIRST_ORDER_MAML=true TUNE=False
```

**24GB GPU（如 RTX 3090）：**
```bash
python ippo_cnn_coop_mining.py NUM_ENVS=512 NUM_STEPS=1000 TUNE=False
python rl2_cnn_coop_mining.py NUM_ENVS=512 RNN_HIDDEN_SIZE=128 TUNE=False
```

**40GB+ GPU（如 A100）：**
```bash
python ippo_cnn_coop_mining.py NUM_ENVS=1024 NUM_STEPS=1000 TUNE=False
python maml_cnn_coop_mining.py NUM_ENVS=512 FIRST_ORDER_MAML=false TUNE=False
```

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
