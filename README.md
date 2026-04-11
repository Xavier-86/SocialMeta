# SocialMeta

> **GPU-Accelerated Multi-Agent RL for Sequential Social Dilemmas**

[![Python](https://img.shields.io/badge/Python-3.10+-2563EB?style=flat&logo=python&logoColor=white)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-0.4.23+-FB8C00?style=flat)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat)](LICENSE)

[English](README.md) · [中文](README.zh.md) · [CLI Guide](CLI_GUIDE.md)

---

## Overview

SocialMeta is a high-performance research framework for Multi-Agent Reinforcement Learning (MARL) and Meta-Reinforcement Learning in Sequential Social Dilemmas (SSDs).

Built on **JAX** with end-to-end JIT compilation for maximum GPU/TPU performance.

---

## Quick Start

```bash
# Install
conda create -n social-meta python=3.10
conda activate social-meta
pip install -r requirements.txt

# Train IPPO
python train.py --algo IPPO --env coop_mining --test

# Train Meta-Learning (RL²)
python train.py --algo RL2 --env coop_mining --test
```

---

## Design Tokens — Configuration Reference

### Environment Matrix

| Environment | Agents | Core Mechanic | Best For |
|-------------|--------|---------------|----------|
| **Cleanup** | 7 | Public goods + pollution | Cooperation under pressure |
| **Coin Game** | 2 | Asymmetric incentives | Conflict resolution |
| **Common Harvest** | 4-7 | Resource depletion | Sustainability |
| **Coop Mining** | 6 | Complementary skills | Skill specialization |
| **Gift** | 2-4 | Reciprocal exchange | Trust building |
| **Mushrooms** | 2 | Risk + uncertainty | Safe exploration |
| **PD Arena** | 4 | Spatial IPD | Emergence of norms |

### Algorithm Capability Matrix

| Algorithm | Type | Environments | Meta-Learning |
|-----------|------|--------------|---------------|
| **IPPO** | MARL | 10 | ❌ |
| **IPPO_raw** | MARL | 10 | ❌ |
| **MAPPO** | MARL | 10 | ❌ |
| **SVO** | Policy Bank | 10 | ❌ |
| **RL2** | Recurrent | 6 | ✅ Trial-based |
| **MAML** | Gradient | 6 | ✅ Gradient-based |

---

## Usage Patterns

### Pattern A: Standard MARL Training

```bash
python train.py \
    --algo IPPO \
    --env coop_mining \
    --num_envs 512 \
    --total_timesteps 3e8
```

### Pattern B: Meta-Learning Training

```bash
# Step 1: Prepare teammate policies
bash get_svo_policies.sh

# Step 2: Train with meta-learning
python train.py \
    --algo RL2 \
    --env coop_mining \
    --trial_episodes 3 \
    --episode_reward_weights "[0.2,0.3,0.5]"
```

### Pattern C: Hyperparameter Sweep

```bash
python train.py \
    --algo IPPO \
    --env coop_mining \
    --tune
```

---

## Configuration Tokens

### Hardware-Optimized Presets

| GPU VRAM | NUM_ENVS | NUM_STEPS | Algorithm |
|----------|----------|-----------|-----------|
| 8 GB | 128 | 512 | IPPO |
| 8 GB | 64 | 384 | MAML (first-order) |
| 24 GB | 512 | 1000 | IPPO |
| 24 GB | 512 | 384 | RL2 |
| 40 GB+ | 1024 | 1000 | Any |

### Core Training Parameters

```yaml
# PPO Base
TOTAL_TIMESTEPS: 3e8        # Training duration
NUM_ENVS: 512               # Parallel environments
NUM_STEPS: 384              # Steps per update
LR: 0.0003                  # Learning rate
GAMMA: 0.99                 # Discount factor
GAE_LAMBDA: 0.95            # GAE parameter
CLIP_EPS: 0.2               # PPO clip range
ENT_COEF: 0.01              # Entropy bonus
VF_COEF: 0.5                # Value loss weight

# Meta-Learning
TRIAL_EPISODES: 3           # Episodes per trial
EPISODE_REWARD_WEIGHTS:     # Reward weighting per episode
  - 0.2
  - 0.3
  - 0.5
```

### Environment Configuration

```yaml
ENV_KWARGS:
  num_agents: 6             # Number of agents
  num_inner_steps: 1000     # Steps per episode
  num_outer_steps: 3        # Episodes per trial
  shared_rewards: false     # Reward structure
  cnn: true                 # Observation type
  jit: true                 # JIT compilation
```

---

## Project Structure

```
socialmeta/
├── train.py                 # ⭐ Unified training interface
├── socialmeta/              # Core library
│   ├── environments/        # 8 SSD environments
│   ├── wrappers/            # Observation wrappers
│   └── registration.py      # Environment registry
├── algorithms/              # MARL implementations
│   ├── IPPO/                # Independent PPO (10 envs)
│   ├── MAPPO/               # Multi-Agent PPO (10 envs)
│   ├── RL2/                 # RL² meta-learning (6 envs)
│   ├── MAML/                # MAML meta-learning (6 envs)
│   └── SVO/                 # SVO policy bank (10 envs)
├── evaluation/              # Cross-play evaluation
├── checkpoints/             # Model checkpoints
└── svo-policies/            # Teammate policy bank
```

---

## CLI Reference

```bash
# Status overview
./socialmeta-cli status

# Train with specific resources
./socialmeta-cli train \
    --algo IPPO \
    --env coop_mining \
    --num-envs 512

# List available resources
./socialmeta-cli list
./socialmeta-cli list --svo
./socialmeta-cli list --checkpoints
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for complete reference.

---

## Citation

```bibtex
@software{socialmeta2025,
  title = {SocialMeta: Meta-Reinforcement Learning for SSDs},
  year = {2025},
  url = {https://github.com/your-repo/socialmeta}
}

@software{socialjax2024,
  title = {SocialJax: JAX-based Multi-Agent RL},
  author = {Social AI Lab},
  year = {2024},
  url = {https://github.com/cooperativex/SocialJax}
}
```

---

## License

MIT License — see [LICENSE](LICENSE)
