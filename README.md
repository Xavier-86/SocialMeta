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
- **🔧 Hydra Configuration**: Flexible, composable experiment configuration
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

### Environment Setup

All algorithms require the project root in `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/socialmeta:$PYTHONPATH
```

Or run from within the algorithm directory with proper path setup.

### Training IPPO (Independent PPO)

IPPO is a standard MARL baseline where each agent learns independently using PPO.

```bash
cd algorithms/IPPO

# Quick test run
python ippo_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=50000 \
    NUM_ENVS=32 \
    WANDB_MODE=disabled \
    TUNE=False

# Full training with wandb logging
python ippo_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=3e8 \
    NUM_ENVS=256 \
    LR=0.0005 \
    TUNE=False

# Train on different environments
python ippo_cnn_cleanup.py TOTAL_TIMESTEPS=3e8 TUNE=False
python ippo_cnn_coins.py TOTAL_TIMESTEPS=3e8 TUNE=False
```

**Key Parameters:**
- `TOTAL_TIMESTEPS`: Total environment steps for training
- `NUM_ENVS`: Number of parallel environments (higher = better GPU utilization)
- `NUM_STEPS`: Steps per update (default: 1000)
- `LR`: Learning rate (default: 0.0005)
- `REWARD`: `"individual"` or `"common"` reward structure
- `TUNE`: Set to `False` for single runs, `True` for hyperparameter sweeps
- `WANDB_MODE`: `"online"`, `"offline"`, or `"disabled"`

### Training RL² (Meta-Learning with Memory)

RL² enables agents to adapt to new teammates during deployment using recurrent memory.

First, download or train SVO teammate policies:

```bash
cd /path/to/socialmeta
bash get_svo_policies.sh
```

Then train RL²:

```bash
cd algorithms/RL2

# Quick test
python rl2_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=50000 \
    NUM_ENVS=32 \
    TRIAL_EPISODES=3 \
    EPISODE_REWARD_WEIGHTS=[0.2,0.3,0.5] \
    WANDB_MODE=disabled \
    TUNE=False

# Full training
python rl2_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=3e8 \
    NUM_ENVS=512 \
    NUM_STEPS=384 \
    TRIAL_EPISODES=3 \
    EPISODE_REWARD_WEIGHTS=[0.2,0.3,0.5] \
    RNN_HIDDEN_SIZE=128 \
    TUNE=False
```

**Meta-Learning Specific Parameters:**
- `TRIAL_EPISODES`: Number of episodes per trial (meta-episode)
- `EPISODE_REWARD_WEIGHTS`: Weighting for episodes in a trial (list length must match `TRIAL_EPISODES`)
- `RNN_HIDDEN_SIZE`: LSTM hidden dimension (default: 128)
- `TEAMMATE_POLICY_DIR`: Directory containing teammate policies
- `EVAL_DURING_TRAIN`: Enable periodic evaluation during training
- `EVAL_TIMES`: Number of evaluation checkpoints during training

### Training MAML (Model-Agnostic Meta-Learning)

MAML learns a good initialization that can quickly adapt to new teammates with gradient updates.

```bash
cd algorithms/MAML

# Quick test (first-order approximation recommended for lower memory)
python maml_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=50000 \
    NUM_ENVS=32 \
    TRIAL_EPISODES=3 \
    EPISODE_REWARD_WEIGHTS=[0.2,0.3,0.5] \
    FIRST_ORDER_MAML=true \
    WANDB_MODE=disabled \
    TUNE=False

# Full training
python maml_cnn_coop_mining.py \
    TOTAL_TIMESTEPS=3e8 \
    NUM_ENVS=512 \
    OUTER_LR=1e-4 \
    INNER_LR=2e-3 \
    INNER_STEPS=1 \
    FIRST_ORDER_MAML=true \
    TUNE=False
```

**MAML-Specific Parameters:**
- `OUTER_LR`: Meta-learning rate for outer loop
- `INNER_LR`: Learning rate for inner loop adaptation
- `INNER_STEPS`: Number of gradient steps in inner loop
- `FIRST_ORDER_MAML`: Use first-order approximation (recommended, saves memory)
- `MAML_NUM_MINIBATCHES`: Gradient accumulation splits for memory efficiency
- `MAML_LOSS_REMAT`: Use gradient checkpointing for lower memory usage

### Training with Hyperparameter Sweeps

Set `TUNE=True` to enable wandb sweeps for hyperparameter tuning:

```bash
python ippo_cnn_coop_mining.py TUNE=True
```

This will run multiple experiments with different learning rates and other hyperparameters as defined in the config.

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
│   ├── IPPO/                   # Independent PPO baseline
│   ├── IPPO_raw/               # IPPO without social value orientation
│   ├── MAPPO/                  # Multi-Agent PPO
│   ├── MAML/                   # Model-Agnostic Meta-Learning
│   ├── PPO/                    # Meta-learning PPO variant
│   ├── RL2/                    # RL² recurrent meta-learning
│   └── SVO/                    # Social Value Orientation policies
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

### Example: Customizing Training

```bash
# Modify multiple parameters
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

### Common Parameters Across All Algorithms

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
python ippo_cnn_coop_mining.py \
    ENTITY=your-username \
    PROJECT=socialmeta-experiments \
    WANDB_TAGS=["baseline","coop_mining"] \
    TUNE=False
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
python ippo_cnn_coop_mining.py SEED=42 TUNE=False

# Multiple seeds (sequential)
for seed in 40 41 42 43 44; do
    python ippo_cnn_coop_mining.py SEED=$seed TUNE=False
done
```

## 💡 Tips for Best Performance

### GPU Utilization

- Increase `NUM_ENVS` to improve GPU utilization (512-1024 recommended for A100)
- Use `ENV_SCAN_UNROLL` and `RNN_SCAN_UNROLL` to tune compilation vs. memory trade-offs

### Meta-Learning Optimization

- Start with `FIRST_ORDER_MAML=True` for MAML (much lower memory usage)
- Ensure teammate policies in `TEAMMATE_POLICY_DIR` have diverse strategies
- Use `EVAL_DURING_TRAIN=True` to monitor generalization during training

### Memory Optimization

For large-scale experiments:

```bash
python maml_cnn_coop_mining.py \
    MAML_LOSS_REMAT=true \
    MAML_NUM_MINIBATCHES=4 \
    ENV_SCAN_UNROLL=2 \
    GAE_SCAN_UNROLL=8
```

### Recommended Configurations by Hardware

**For 8GB GPU (e.g., RTX 3070):**
```bash
python ippo_cnn_coop_mining.py NUM_ENVS=128 NUM_STEPS=512 TUNE=False
python maml_cnn_coop_mining.py NUM_ENVS=64 FIRST_ORDER_MAML=true TUNE=False
```

**For 24GB GPU (e.g., RTX 3090):**
```bash
python ippo_cnn_coop_mining.py NUM_ENVS=512 NUM_STEPS=1000 TUNE=False
python rl2_cnn_coop_mining.py NUM_ENVS=512 RNN_HIDDEN_SIZE=128 TUNE=False
```

**For 40GB+ GPU (e.g., A100):**
```bash
python ippo_cnn_coop_mining.py NUM_ENVS=1024 NUM_STEPS=1000 TUNE=False
python maml_cnn_coop_mining.py NUM_ENVS=512 FIRST_ORDER_MAML=false TUNE=False
```

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
