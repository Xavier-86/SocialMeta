# SocialMeta

**Fast, GPU-Accelerated Multi-Agent Reinforcement Learning for Sequential Social Dilemmas with Meta-Learning Support**

[English](README.md) | [中文](README.zh.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.23+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SocialMeta is a high-performance benchmark suite built on [JAX](https://github.com/google/jax) for evaluating multi-agent reinforcement learning (MARL) and meta-reinforcement learning (meta-RL) algorithms in Sequential Social Dilemmas (SSDs). Built upon the excellent [SocialJax](https://github.com/cooperativex/SocialJax) foundation, SocialMeta extends the original framework with comprehensive meta-learning capabilities and enhanced evaluation tools.

## 🚀 Key Features

- **⚡ Pure JAX Implementation**: End-to-end JIT compilation for maximum performance on GPU/TPU
- **🎯 Meta-Learning Ready**: Built-in support for RL² and MAML meta-learning algorithms
- **🤝 Diverse SSD Environments**: 8 challenging multi-agent social dilemmas
- **📊 Comprehensive Evaluation**: Tools for assessing policy generalization against diverse teammates
- **🔧 Unified Interface**: Single command to train any algorithm on any environment
- **🔌 Hydra Configuration**: Flexible, composable experiment configuration
- **📈 W&B Integration**: Built-in experiment tracking and visualization

## 📋 Requirements

- Python >= 3.10
- JAX >= 0.4.23 (with CUDA support for GPU training)
- See `requirements.txt` for full dependencies

## 🔧 Installation

### Using Conda (Recommended)

```bash
# Create and activate environment
conda create -n social-meta python=3.10
conda activate social-meta

# Clone the repository
git clone <repository-url>
cd socialmeta

# Install dependencies
pip install -r requirements.txt
```

### Using Poetry

```bash
# Install dependencies
poetry install

# Activate shell
poetry shell
```

### Verify Installation

```bash
# Test JAX GPU support
python -c "import jax; print(jax.devices())"

# Test environment loading
python -c "import socialmeta; env = socialmeta.make('coop_mining'); print('Environment loaded successfully!')"
```

## 🎮 Environments

SocialMeta includes 8 sequential social dilemma environments:

| Environment | Description | Agents | Challenge |
|-------------|-------------|--------|-----------|
| **Cleanup** | Clean pollution while avoiding over-harvesting | 7 | Public goods dilemma |
| **Coin Game** | Collect coins with asymmetric incentives | 2 | Conflict of interest |
| **Common Harvest** | Sustainable resource management | 4-7 | Tragedy of the commons |
| **Coop Mining** | Mine iron and gold collaboratively | 6 | Cooperation under uncertainty |
| **Gift** | Exchange gifts to maximize welfare | 2-4 | Reciprocity |
| **Mushrooms** | Forage mushrooms with toxicity risk | 2 | Risk assessment |
| **PD Arena** | Spatial iterated prisoner's dilemma | 2-8 | Emergence of cooperation |
| **Territory** | Claim and defend territory | 2-4 | Competition vs. coordination |

All environments support:
- CNN observation processing (grid-based)
- Individual or shared rewards
- Configurable episode lengths
- JIT-compiled step functions

## 🏃 Quick Start

### Unified Training Interface

SocialMeta provides a unified command-line interface for training all algorithms:

```bash
# List all supported algorithms
python train.py --list_algos

# List environments supported by an algorithm
python train.py --algo IPPO --list_envs
```

### Training Examples

#### IPPO (Independent PPO)

```bash
# Quick test run
python train.py --algo IPPO --env coop_mining --test

# Full training
python train.py --algo IPPO --env coop_mining \
    --total_timesteps 3e8 \
    --num_envs 256 \
    --lr 0.0005

# Train on different environments
python train.py --algo IPPO --env cleanup --test
python train.py --algo IPPO --env coins --test
```

#### RL² (Meta-Learning with Memory)

First, download or train SVO teammate policies:

```bash
bash get_svo_policies.sh
```

Then train RL²:

```bash
# Quick test
python train.py --algo RL2 --env coop_mining --test

# Full training
python train.py --algo RL2 --env coop_mining \
    --total_timesteps 3e8 \
    --num_envs 512 \
    --trial_episodes 3 \
    --episode_reward_weights "[0.2,0.3,0.5]"
```

#### MAML (Model-Agnostic Meta-Learning)

```bash
# Quick test (first-order approximation recommended for lower memory)
python train.py --algo MAML --env coop_mining --test --first_order_maml

# Full training
python train.py --algo MAML --env coop_mining \
    --total_timesteps 3e8 \
    --num_envs 512 \
    --lr 1e-4 \
    --first_order_maml
```

#### Other Algorithms

```bash
# MAPPO
python train.py --algo MAPPO --env coop_mining --test

# IPPO without SVO wrapper
python train.py --algo IPPO_raw --env coop_mining --test
```

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--algo` | Algorithm: IPPO, IPPO_raw, MAPPO, RL2, MAML, SVO | Required |
| `--env` | Environment name | Required |
| `--total_timesteps` | Total training steps | 50000 |
| `--num_envs` | Parallel environments | 32 |
| `--num_steps` | Steps per update | 384-1000 |
| `--lr` | Learning rate | 0.0003-0.0005 |
| `--seed` | Random seed | 30 |
| `--wandb_mode` | W&B logging: online/offline/disabled | disabled |
| `--tune` | Enable hyperparameter sweep | False |
| `--test` | Quick test mode (1000 steps) | False |

### Meta-Learning Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--trial_episodes` | Episodes per trial | 3 |
| `--episode_reward_weights` | Reward weighting per episode | [0.2,0.3,0.5] |
| `--first_order_maml` | Use first-order MAML | False |

### Training with Hyperparameter Sweeps

Enable wandb sweeps for hyperparameter tuning:

```bash
python train.py --algo IPPO --env coop_mining --tune
```

## 📊 Evaluation

Evaluate trained policies against diverse teammate strategies:

```bash
cd evaluation/coop_mining

# Evaluate a trained checkpoint
python evaluate.py \
    --checkpoint /path/to/checkpoint.pkl \
    --num_episodes 100
```

Evaluation supports:
- Cross-play with SVO-trained policies
- Generalization testing with held-out teammates
- Performance metrics across different teammate configurations

## 🏗️ Project Structure

```
socialmeta/
├── train.py                    # Unified training interface ⭐
├── socialmeta/                 # Core library
│   ├── environments/           # SSD environment implementations
│   │   ├── cleanup/
│   │   ├── coin_game/
│   │   ├── common_harvest/
│   │   ├── coop_mining/
│   │   ├── gift/
│   │   ├── mushrooms/
│   │   ├── pd_arena/
│   │   └── territory/
│   ├── wrappers/               # Environment wrappers
│   └── registration.py         # Environment registry
│
├── algorithms/                 # MARL & Meta-RL implementations
│   ├── IPPO/                   # Independent PPO baseline (10 envs)
│   ├── IPPO_raw/               # IPPO without SVO wrapper (10 envs)
│   ├── MAPPO/                  # Multi-Agent PPO (10 envs)
│   ├── MAML/                   # Model-Agnostic Meta-Learning (1 env)
│   ├── PPO/                    # Meta-learning PPO variant (1 env)
│   ├── RL2/                    # RL² recurrent meta-learning (1 env)
│   └── SVO/                    # Social Value Orientation policies (10 envs)
│
├── evaluation/                 # Evaluation scripts
│   ├── cleanup/
│   ├── coin/
│   ├── coop_mining/
│   ├── gift/
│   ├── harvest_closed/
│   ├── harvest_partnership/
│   ├── mushroom/
│   └── pd_arena/
│
├── checkpoints/                # Saved model checkpoints
├── fixed_policy/               # Scripted baseline policies
└── speed_test/                 # Performance benchmarks
```

## ⚙️ Configuration

All algorithms use [Hydra](https://hydra.cc/) for configuration management. Configs are stored in `algorithms/<ALGO>/config/`.

### Algorithm-Environment Compatibility

| Algorithm | Supported Environments | Count |
|-----------|------------------------|-------|
| IPPO | coop_mining, cleanup, coins, gift, mushrooms, pd_arena, harvest_common, harvest_common_closed, harvest_common_partnership, territory_open | 10 |
| IPPO_raw | Same as IPPO | 10 |
| MAPPO | coop_mining, cleanup, coins, gifts, mushrooms, pd_arena, harvest_common, harvest_common_closed, harvest_common_partnership, territory_open | 10 |
| RL² | coop_mining | 1 |
| MAML | coop_mining | 1 |
| SVO | coop_mining, cleanup, coin, gift, mushroom, pd_arena, harvest_open, harvest_closed, harvest_partnership, territory_open | 10 |

### Advanced Configuration via Hydra

For advanced use cases, you can still call algorithm scripts directly:

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

### Environment Configuration

Each environment accepts `ENV_KWARGS`:

```yaml
ENV_KWARGS:
  num_agents: 6              # Number of agents
  num_inner_steps: 1000      # Steps per episode
  num_outer_steps: 3         # Episodes per trial (meta-learning)
  shared_rewards: false      # Individual vs shared rewards
  cnn: true                  # Use CNN observations
  jit: true                  # JIT-compile environment
```

### PPO Hyperparameters (All Algorithms)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TOTAL_TIMESTEPS` | Total training steps | 3e8 |
| `NUM_ENVS` | Parallel environments | 256-512 |
| `NUM_STEPS` | Steps per update | 384-1000 |
| `LR` / `OUTER_LR` | Learning rate | 0.0003-0.0005 |
| `GAMMA` | Discount factor | 0.99 |
| `GAE_LAMBDA` | GAE lambda | 0.95 |
| `CLIP_EPS` | PPO clip epsilon | 0.2 |
| `ENT_COEF` | Entropy coefficient | 0.01 |
| `VF_COEF` | Value function coefficient | 0.5 |
| `MAX_GRAD_NORM` | Gradient clipping | 0.5 |
| `ACTIVATION` | Activation function | "relu" |
| `ANNEAL_LR` | Anneal learning rate | True |
| `SEED` | Random seed | 30 |
| `TUNE` | Enable hyperparameter sweep | False |
| `WANDB_MODE` | W&B logging mode | "online" |

## 📈 Experiment Tracking

SocialMeta integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking:

```bash
# Login to W&B
wandb login

# Training with W&B logging
python train.py --algo IPPO --env coop_mining \
    --wandb_mode online
```

Logged metrics include:
- Episode returns (individual and team)
- Policy entropy
- Value function loss
- Evaluation performance (for meta-learning)
- Training throughput (steps/sec)

## 🧪 Reproducibility

Set random seeds for reproducible experiments:

```bash
# Single seed
python train.py --algo IPPO --env coop_mining --seed 42

# Multiple seeds (sequential)
for seed in 40 41 42 43 44; do
    python train.py --algo IPPO --env coop_mining --seed $seed
done
```

## 💡 Tips for Best Performance

### GPU Utilization

- Increase `--num_envs` to improve GPU utilization (512-1024 recommended for A100)
- Use `ENV_SCAN_UNROLL` and `RNN_SCAN_UNROLL` to tune compilation vs. memory trade-offs

### Meta-Learning Optimization

- Start with `--first_order_maml` for MAML (much lower memory usage)
- Ensure teammate policies in `TEAMMATE_POLICY_DIR` have diverse strategies
- Use `EVAL_DURING_TRAIN=True` to monitor generalization during training

### Memory Optimization

For large-scale experiments:

```bash
python train.py --algo MAML --env coop_mining \
    --first_order_maml
```

Or using direct script access:

```bash
cd algorithms/MAML
python maml_cnn_coop_mining.py \
    MAML_LOSS_REMAT=true \
    MAML_NUM_MINIBATCHES=4 \
    ENV_SCAN_UNROLL=2 \
    GAE_SCAN_UNROLL=8
```

### Recommended Configurations by Hardware

**For 8GB GPU (e.g., RTX 3070):**
```bash
python train.py --algo IPPO --env coop_mining --num_envs 128 --num_steps 512
python train.py --algo MAML --env coop_mining --num_envs 64 --first_order_maml
```

**For 24GB GPU (e.g., RTX 3090):**
```bash
python train.py --algo IPPO --env coop_mining --num_envs 512 --num_steps 1000
python train.py --algo RL2 --env coop_mining --num_envs 512
```

**For 40GB+ GPU (e.g., A100):**
```bash
python train.py --algo IPPO --env coop_mining --num_envs 1024 --num_steps 1000
python train.py --algo MAML --env coop_mining --num_envs 512
```

## 🐛 Known Issues

| Issue | Affected | Workaround |
|-------|----------|------------|
| SVO shape error | SVO algorithm | Use IPPO_raw instead |
| Territory attribute error | Territory environment | Use other environments |
| MAPPO GIF save error | MAPPO evaluation | Training works fine, GIF is optional |
## 📚 Citation

If you use SocialMeta in your research, please cite both this work and the original SocialJax:

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

SocialMeta is built upon the excellent [SocialJax](https://github.com/cooperativex/SocialJax) codebase. We thank the original authors for providing a high-quality foundation for multi-agent RL research in social dilemmas.

Additional inspiration from:
- [PureJaxRL](https://github.com/luchris429/purejaxrl) for JAX-native RL implementations
- [JaxMARL](https://github.com/FLAIROx/JaxMARL) for multi-agent environment designs
