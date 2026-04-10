# SocialMeta CLI Usage Guide

Unified command-line tool for training, managing, and monitoring multi-agent reinforcement learning models in the SocialMeta project.

## Installation

```bash
# Add execute permission to the script
chmod +x socialmeta-cli

# (Optional) Add to PATH
ln -s $(pwd)/socialmeta-cli ~/.local/bin/socialmeta-cli
```

## Quick Start

```bash
# Check training status
./socialmeta-cli status

# Train SVO policy
./socialmeta-cli train --algo SVO --env cleanup --num-envs 64

# Train IPPO
./socialmeta-cli train --algo IPPO --env coop_mining

# List all SVO policies
./socialmeta-cli list --svo
```

## Command Reference

### 1. status - Check Training Status

```bash
./socialmeta-cli status
```

Displays:
- SVO policy completion status (7/7 angles per environment)
- Checkpoints statistics
- Running training tasks

### 2. train - Train Models

```bash
./socialmeta-cli train --algo <algorithm> --env <environment> [options]
```

#### Supported Algorithms

| Algorithm | Supported Environments |
|-----------|------------------------|
| SVO | cleanup, coin, coop_mining, gift, mushroom, pd_arena, harvest_open, harvest_closed, harvest_partnership |
| IPPO | coop_mining, cleanup, coins, gift, mushrooms, pd_arena, harvest_common, harvest_common_closed, harvest_common_partnership, territory_open |
| IPPO_raw | Same as IPPO |
| MAPPO | Same as IPPO |
| RL2 | coop_mining |
| MAML | coop_mining |

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--algo` | Algorithm name (required) | - |
| `--env` | Environment name (required) | - |
| `--num-envs` | Number of parallel environments | Auto-detect |
| `--seed` | Random seed | 30 |
| `--wandb` | Enable W&B logging | Disabled |
| `--test` | Test mode (short training) | No |

#### Examples

```bash
# Train SVO - cleanup (use 64 parallel environments for limited VRAM)
./socialmeta-cli train --algo SVO --env cleanup --num-envs 64

# Train IPPO - coop_mining with W&B
./socialmeta-cli train --algo IPPO --env coop_mining --wandb

# Train MAPPO - harvest (test mode)
./socialmeta-cli train --algo MAPPO --env harvest_common --test
```

### 3. list - List Resources

```bash
# List all supported algorithms and environments
./socialmeta-cli list

# List all SVO policies with file sizes
./socialmeta-cli list --svo

# List all checkpoints
./socialmeta-cli list --checkpoints
```

### 4. clean - Clean Temporary Files

```bash
./socialmeta-cli clean
```

Cleans:
- `outputs/` - Hydra output directory
- `wandb/` - W&B cache (only when files are numerous)

## Training Recommendations

### VRAM Configuration (8GB GPU)

| Environment | Agents | Recommended NUM_ENVS |
|-------------|--------|---------------------|
| coin | 2 | 256-430 |
| gift/mushroom | 5 | 128-172 |
| pd_arena | 4 | 160-215 |
| cleanup/harvest | 7 | 90-123 |
| coop_mining | 6 | 128 |

### SVO Policy Training Order

Completed ✅ (63/70):
- cleanup, coin, coop_mining, gift, mushroom, pd_arena
- harvest_open, harvest_closed, harvest_partnership

Skipped ❌ (0/7):
- territory_open (environment doesn't support SVO)

## File Structure

```
svo-policies/
├── cleanup/
│   ├── svo_angle_0.pkl
│   ├── svo_angle_15.pkl
│   └── ... (7 angles total)
├── coin/
└── ... (other environments)

checkpoints/
├── individual/     # IPPO standard policies
├── maml/           # MAML meta-learning
└── rl2/            # RL² meta-learning
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce number of parallel environments
./socialmeta-cli train --algo SVO --env cleanup --num-envs 32
```

### Checkpoint Not Saved

Check `checkpoints/` and `svo-policies/` directories:
```bash
./socialmeta-cli list --checkpoints
./socialmeta-cli list --svo
```

### Environment Not Found

View supported environments list:
```bash
./socialmeta-cli list
```

## Migration from Old Scripts

The following old scripts have been replaced by the CLI:

| Old Script | CLI Equivalent |
|------------|----------------|
| `train_all_svo_policies.sh` | `./socialmeta-cli train --algo SVO --env <env>` |
| `train_svo_with_memory_control.sh` | `./socialmeta-cli train --algo SVO --env <env> --num-envs 64` |
| `train_svo_cleanup_low_mem.sh` | `./socialmeta-cli train --algo SVO --env cleanup --num-envs 64` |
| `get_svo_policies.sh` | Kept (download pre-trained models) |

## Advanced Usage

### Batch Training

```bash
# Batch train all harvest environments
for env in harvest_open harvest_closed harvest_partnership; do
    ./socialmeta-cli train --algo SVO --env $env --num-envs 64
done
```

### Multi-Seed Training

```bash
# Train with different seeds
for seed in 30 42 123; do
    ./socialmeta-cli train --algo IPPO --env coop_mining --seed $seed
done
```

## Related Documentation

- `README.md` - Project overview (English)
- `README.zh.md` - Project overview (Chinese)
- `AGENTS.md` - Detailed project documentation
