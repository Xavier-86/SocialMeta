# SocialMeta CLI

> Unified command-line interface for training, managing, and monitoring multi-agent RL experiments.

## Installation

```bash
# Make executable
chmod +x sm

# (Optional) Create alias or symlink
ln -s $(pwd)/./cli ~/.local/bin/sm
```

## Commands

| Command | Description |
|---------|-------------|
| `./cli train` | Train models with full parameter control |
| `./cli status` | View training status and resource overview |
| `./cli list` | List algorithms, environments, checkpoints |
| `./cli clean` | Clean temporary files and caches |

---

## Train

Unified training interface for all algorithms.

```bash
./cli train --algo <ALGORITHM> --env <ENVIRONMENT> [options]
```

### Quick Start

```bash
# Train IPPO
./cli train --algo IPPO --env coop_mining

# Train with custom parameters
./cli train --algo IPPO --env coop_mining --num_envs 512 --lr 0.0005

# Meta-learning with RL²
./cli train --algo RL2 --env coop_mining --trial_episodes 3

# Quick test (1000 steps)
./cli train --algo IPPO --env coop_mining --test
```

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--algo` | string | required | Algorithm: IPPO, IPPO_raw, MAPPO, RL2, MAML, SVO |
| `--env` | string | required | Environment name |
| `--total_timesteps` | float | 50000 | Total training steps |
| `--num_envs` | int | 32 | Parallel environments |
| `--num_steps` | int | auto | Steps per update |
| `--lr` | float | auto | Learning rate |
| `--seed` | int | 30 | Random seed |
| `--wandb_mode` | string | disabled | W&B logging: online/offline/disabled |
| `--tune` | flag | false | Enable hyperparameter sweep |
| `--test` | flag | false | Quick test mode |

### Meta-Learning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--trial_episodes` | int | 3 | Episodes per trial (RL2/MAML) |
| `--episode_reward_weights` | string | auto | Reward weights, e.g., "[0.2,0.3,0.5]" |
| `--first_order_maml` | flag | false | Use first-order MAML approximation |

---

## Status

Display comprehensive training status.

```bash
./cli status
```

Shows:
- SVO policy completion (7 angles per environment)
- Checkpoint counts by algorithm
- Active training processes

---

## List

List available resources.

```bash
# List all algorithms
./cli list

# List environments for specific algorithm
./cli list --algo IPPO

# List SVO policies with sizes
./cli list --svo

# List all checkpoints
./cli list --checkpoints
```

---

## Clean

Clean temporary files and caches.

```bash
./cli clean
```

Removes:
- `outputs/` - Hydra output directories
- `wandb/` - W&B cache (if >100 files)

---

## Algorithm-Environment Matrix

| Algorithm | Type | Environments | Meta-Learning |
|-----------|------|--------------|---------------|
| **IPPO** | MARL | 10 | ❌ |
| **IPPO_raw** | MARL | 10 | ❌ |
| **MAPPO** | MARL | 10 | ❌ |
| **SVO** | Policy Bank | 9 | ❌ |
| **RL2** | Recurrent | 6 | ✅ Trial-based |
| **MAML** | Gradient | 6 | ✅ Gradient-based |

### Supported Environments

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

## VRAM-Optimized Configurations

| GPU | Environment | Agents | Recommended NUM_ENVS |
|-----|-------------|--------|---------------------|
| 8GB | coin | 2 | 256 |
| 8GB | gift | 5 | 128 |
| 8GB | pd_arena | 4 | 160 |
| 8GB | cleanup | 7 | 90 |
| 8GB | coop_mining | 6 | 128 |
| 24GB | any | varies | 512 |
| 40GB+ | any | varies | 1024 |

---

## Troubleshooting

### Out of Memory

```bash
# Reduce parallel environments
./cli train --algo SVO --env cleanup --num_envs 32
```

### Environment Not Found

```bash
# Check supported combinations
./cli list --algo IPPO
```

### Checkpoint Issues

```bash
# Verify checkpoint directories
./cli list --checkpoints
./cli list --svo
```

---

## Related Documentation

- [README.md](README.md) — Project overview
- [README.zh.md](README.zh.md) — 中文项目概览
- [docs/](docs/) — Detailed documentation
