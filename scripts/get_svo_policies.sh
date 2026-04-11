#!/usr/bin/env bash

#get_svo_policies.sh --seed 42 --angles "0,15,30,45,60,75,90" --wandb-mode disabled
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-30}"
WANDB_MODE="${WANDB_MODE:-online}"

# Default angles
ANGLES_STR="${ANGLES:-0 15 30 45 60 75 90}"

usage() {
  cat <<'EOF'
Usage: bash get_svo_policies.sh [--seed N] [--angles "0 15 30 ..."] [--wandb-mode MODE]

Runs all SVO scripts in non-tune mode (TUNE=False) for a fixed seed and a list of SVO angles.

Options:
  --seed N            Default: 30
  --angles "..."      Default: "0 15 30 45 60 75 90" (spaces or commas both ok)
  --wandb-mode MODE   Default: online (e.g. online|offline|disabled)

Env vars (optional):
  PYTHON_BIN, SEED, ANGLES, WANDB_MODE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="${2:?missing seed}"
      shift 2
      ;;
    --angles)
      ANGLES_STR="${2:?missing angles}"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="${2:?missing wandb mode}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Support comma-separated angles too
ANGLES_STR="${ANGLES_STR//,/ }"
read -r -a ANGLES <<<"$ANGLES_STR"

scripts=(
  # "algorithms/SVO/svo_cnn_coin.py"
  # "algorithms/SVO/svo_cnn_gift.py"
  # "algorithms/SVO/svo_cnn_cleanup.py"
  "algorithms/SVO/svo_cnn_coop_mining.py" #
  # "algorithms/SVO/svo_cnn_mushroom.py"
  # "algorithms/SVO/svo_cnn_pd_arena.py"
  "algorithms/SVO/svo_cnn_territory_open.py" #
  # "algorithms/SVO/svo_cnn_harvest_open.py"
  "algorithms/SVO/svo_cnn_harvest_closed.py" #
  "algorithms/SVO/svo_cnn_harvest_partnership.py" #
)

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

for script in "${scripts[@]}"; do
  for angle in "${ANGLES[@]}"; do
    echo "==> $script (seed=$SEED angle=$angle wandb=$WANDB_MODE)"
    set +e
    "$PYTHON_BIN" "$script" \
      TUNE=False \
      SEED="$SEED" \
      WANDB_MODE="$WANDB_MODE" \
      ENV_KWARGS.svo_ideal_angle_degrees="$angle"
    status=$?
    set -e
    if [[ $status -ne 0 ]]; then
      echo "!! failed: $script (angle=$angle) exit=$status" >&2
    fi
  done
done
