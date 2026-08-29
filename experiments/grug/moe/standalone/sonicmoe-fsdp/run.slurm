#!/bin/bash
# Launcher for the sonicmoe-fsdp benchmarks (smoe env, 1 node / 4x GB200).
#   sbatch run.slurm <SCRIPT.py> <NPROC> [args...]
# NPROC=1 -> plain python (single-GPU probe); NPROC>1 -> torchrun.
# The winning comm path (sonicmoe's Triton symm-mem all-gather) never touches NCCL tuning knobs;
# NCCL_NET_PLUGIN=none below is the only NCCL env var this launcher sets.
#SBATCH --job-name=smf
#SBATCH -p batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/wentaog/ep_bench/srun/sonicmoe_fsdp/logs/%x_%j.log
#SBATCH --error=/lustre/fsw/portfolios/nemotron/users/wentaog/ep_bench/srun/sonicmoe_fsdp/logs/%x_%j.log

set -uo pipefail
SCRIPT="${1:?usage: run.slurm SCRIPT NPROC [args...]}"
NPROC="${2:-4}"
shift $(( $# < 2 ? $# : 2 )) 2>/dev/null || true
ARGS=("$@")

ROOT=/lustre/fsw/portfolios/nemotron/users/wentaog/ep_bench/srun/sonicmoe_fsdp
DIR=/home/wentaog/marin/experiments/grug/moe/standalone/sonicmoe-fsdp
mkdir -p "$ROOT/logs" "$ROOT/cute-dsl-cache" "$ROOT/triton-cache"

echo "host=$(hostname) job=$SLURM_JOB_ID script=$SCRIPT nproc=$NPROC args=${ARGS[*]:-<none>}"
nvidia-smi -L
source /home/wentaog/miniforge3/etc/profile.d/conda.sh
conda activate smoe
echo "python: $(command -v python)  torch: $(python -c 'import torch;print(torch.__version__)')"

export CUTE_DSL_CACHE_DIR="$ROOT/cute-dsl-cache"
export TRITON_CACHE_DIR="$ROOT/triton-cache"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_NET_PLUGIN=none

cd "$DIR"
if [ "$NPROC" = "1" ]; then
  python "$SCRIPT" "${ARGS[@]}"
else
  torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT" "${ARGS[@]}"
fi
RC=$?
echo "EXIT_CODE=$RC"
exit $RC
