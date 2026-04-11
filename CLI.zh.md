# SocialMeta CLI

> 统一的命令行工具，用于训练、管理和监控多智能体强化学习实验。

## 安装

```bash
# 添加执行权限
chmod +x sm

# (可选) 创建别名或软链接
ln -s $(pwd)/./cli ~/.local/bin/sm
```

## 命令

| 命令 | 说明 |
|------|------|
| `./cli train` | 训练模型，支持完整参数控制 |
| `./cli status` | 查看训练状态和资源概览 |
| `./cli list` | 列出算法、环境、检查点 |
| `./cli clean` | 清理临时文件和缓存 |

---

## Train

所有算法的统一训练接口。

```bash
./cli train --algo <算法> --env <环境> [参数]
```

### 快速开始

```bash
# 训练 IPPO
./cli train --algo IPPO --env coop_mining

# 自定义参数训练
./cli train --algo IPPO --env coop_mining --num_envs 512 --lr 0.0005

# 元学习 RL²
./cli train --algo RL2 --env coop_mining --trial_episodes 3

# 快速测试 (1000 步)
./cli train --algo IPPO --env coop_mining --test
```

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--algo` | string | 必需 | 算法: IPPO, IPPO_raw, MAPPO, RL2, MAML, SVO |
| `--env` | string | 必需 | 环境名称 |
| `--total_timesteps` | float | 50000 | 总训练步数 |
| `--num_envs` | int | 32 | 并行环境数 |
| `--num_steps` | int | 自动 | 每次更新步数 |
| `--lr` | float | 自动 | 学习率 |
| `--seed` | int | 30 | 随机种子 |
| `--wandb_mode` | string | disabled | W&B 日志: online/offline/disabled |
| `--tune` | flag | false | 启用超参数搜索 |
| `--test` | flag | false | 快速测试模式 |

### 元学习参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--trial_episodes` | int | 3 | 每 trial 的回合数 (RL2/MAML) |
| `--episode_reward_weights` | string | 自动 | 奖励权重, 如 "[0.2,0.3,0.5]" |
| `--first_order_maml` | flag | false | 使用一阶 MAML 近似 |

---

## Status

显示综合训练状态。

```bash
./cli status
```

显示内容:
- SVO 策略完成状态 (每环境 7 个角度)
- 各算法的检查点数量
- 活跃的训练进程

---

## List

列出可用资源。

```bash
# 列出所有算法
./cli list

# 列出特定算法支持的环境
./cli list --algo IPPO

# 列出 SVO 策略及大小
./cli list --svo

# 列出所有检查点
./cli list --checkpoints
```

---

## Clean

清理临时文件和缓存。

```bash
./cli clean
```

清理内容:
- `outputs/` - Hydra 输出目录
- `wandb/` - W&B 缓存 (超过 100 个文件时)

---

## 算法-环境矩阵

| 算法 | 类型 | 环境数 | 元学习 |
|------|------|--------|--------|
| **IPPO** | MARL | 10 | ❌ |
| **IPPO_raw** | MARL | 10 | ❌ |
| **MAPPO** | MARL | 10 | ❌ |
| **SVO** | 策略库 | 9 | ❌ |
| **RL2** | 循环网络 | 6 | ✅ 基于 Trial |
| **MAML** | 梯度 | 6 | ✅ 基于梯度 |

### 支持的环境

**IPPO/IPPO_raw/MAPPO:**
```
coop_mining, cleanup, coins, gift, mushrooms, pd_arena,
```

**RL2/MAML:**
```
coop_mining, cleanup, coins, gift, mushrooms, pd_arena
```

**SVO:**
```
coop_mining, cleanup, coin, gift, mushroom, pd_arena,
harvest_open, harvest_closed, harvest_partnership
```

---

## VRAM 优化配置

| GPU | 环境 | 智能体数 | 推荐的 NUM_ENVS |
|-----|------|----------|----------------|
| 8GB | coin | 2 | 256 |
| 8GB | gift | 5 | 128 |
| 8GB | pd_arena | 4 | 160 |
| 8GB | cleanup | 7 | 90 |
| 8GB | coop_mining | 6 | 128 |
| 24GB | 任意 | 可变 | 512 |
| 40GB+ | 任意 | 可变 | 1024 |

---

## 故障排除

### 显存不足

```bash
# 减少并行环境数
./cli train --algo SVO --env cleanup --num_envs 32
```

### 环境未找到

```bash
# 检查支持的组合
./cli list --algo IPPO
```

### 检查点问题

```bash
# 验证检查点目录
./cli list --checkpoints
./cli list --svo
```

---

## 相关文档

- [README.md](README.md) — 英文项目概览
- [README.zh.md](README.zh.md) — 中文项目概览
- [docs/](docs/) — 详细文档
