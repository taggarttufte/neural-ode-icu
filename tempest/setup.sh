#!/bin/bash
# Run this once on Tempest to create your Python environment.
# Ask UIT for the correct module names before running.

module load python/3.11    # confirm version with UIT
module load cuda/12.x      # confirm version with UIT

# Create virtual environment in your home directory
python -m venv ~/envs/neural-ode
source ~/envs/neural-ode/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torchdiffeq numpy pandas scikit-learn matplotlib tqdm xgboost

echo "Environment ready. Activate with:"
echo "  source ~/envs/neural-ode/bin/activate"
