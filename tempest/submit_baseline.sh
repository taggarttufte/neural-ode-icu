#!/bin/bash
#SBATCH --job-name=xgboost-baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=02:00:00                  # 2 hours — XGBoost is fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # XGBoost benefits from many cores
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                     # optional — XGBoost GPU support
# #SBATCH --partition=gpu               # uncomment and set per UIT guidance

# ── Environment setup ─────────────────────────────────────────────────────
module load python/3.11
module load cuda/12.x

source ~/envs/neural-ode/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────────
NETID="your_netid"
PROJECT_DIR="/home/${NETID}/neural-ode-icu"
DATA_DIR="/scratch/${NETID}/mimic-iv/processed"
OUTPUT_DIR="/scratch/${NETID}/results/xgboost"

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# ── Run ───────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"

python ${PROJECT_DIR}/tempest/baseline_hpc.py \
    --data-dir   ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR}

echo "Job finished: $(date)"
