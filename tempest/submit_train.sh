#!/bin/bash
#SBATCH --job-name=neural-ode-icu
#SBATCH --output=logs/train_%j.out       # stdout log (%j = job ID)
#SBATCH --error=logs/train_%j.err        # stderr log
#SBATCH --time=08:00:00                  # max wall time (8 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # for num_workers=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                     # request 1 GPU
# NOTE: ask UIT which partition/GPU type to use on Tempest
# Common options: --partition=gpu  or  --partition=icas-gpu
# e.g. #SBATCH --partition=gpu

# ── Environment setup ─────────────────────────────────────────────────────
# Load modules (ask UIT for exact module names on Tempest)
module load python/3.11      # or whatever version UIT recommends
module load cuda/12.x        # match your PyTorch CUDA version

# Activate virtual environment (create with setup.sh first)
source ~/envs/neural-ode/bin/activate

# ── Paths (update NETID and paths to match your Tempest setup) ─────────────
NETID="your_netid"
PROJECT_DIR="/home/${NETID}/neural-ode-icu"
DATA_DIR="/scratch/${NETID}/mimic-iv/processed"    # secure storage location
OUTPUT_DIR="/scratch/${NETID}/results/neural-ode"

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# ── Run ───────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python ${PROJECT_DIR}/tempest/train_hpc.py \
    --data-dir   ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs 50 \
    --batch-size 64 \
    --latent-dim 32 \
    --hidden-dim 64 \
    --kl-weight 0.1 \
    --early-stop-patience 10

echo "Job finished: $(date)"
