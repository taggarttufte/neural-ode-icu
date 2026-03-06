# Tempest HPC — Setup & Usage Guide

This folder contains HPC-ready versions of the training scripts and SLURM job
submission files for running on MSU's Tempest cluster.

## Files

| File | Purpose |
|------|---------|
| `train_hpc.py` | Neural ODE training (Linux, num_workers=4, configurable paths) |
| `baseline_hpc.py` | XGBoost baseline (GPU-enabled, configurable paths) |
| `submit_train.sh` | SLURM job script for Neural ODE training |
| `submit_baseline.sh` | SLURM job script for XGBoost baseline |
| `setup.sh` | One-time environment setup |

## Key differences from local version

| Feature | Local (Windows) | HPC (Tempest) |
|---------|----------------|---------------|
| `num_workers` | 0 (Windows bug) | 4 (Linux works) |
| Data paths | Hardcoded `data/` folder | `--data-dir` argument |
| Output paths | Hardcoded `results/` folder | `--output-dir` argument |
| XGBoost device | CPU | GPU if available |
| Job submission | Direct `python` command | `sbatch submit_*.sh` |

## First-time setup on Tempest

**1. SSH in**
```bash
ssh NETID@tempest.montana.edu
```

**2. Clone the repo**
```bash
cd ~
git clone https://github.com/taggarttufte/neural-ode-icu.git
```

**3. Set up Python environment** (confirm module names with UIT first)
```bash
cd neural-ode-icu/tempest
bash setup.sh
```

**4. Update NETID and paths in job scripts**

Edit `submit_train.sh` and `submit_baseline.sh`:
- Replace `your_netid` with your MSU NetID
- Confirm `DATA_DIR` points to where MIMIC-IV is stored on secure storage
- Confirm partition name with UIT (`--partition=gpu` or similar)

## Running jobs

```bash
# Submit Neural ODE training (~4-8 hrs on MIMIC-IV ICU)
sbatch tempest/submit_train.sh

# Submit XGBoost baseline (~30-60 min on MIMIC-IV ICU)
sbatch tempest/submit_baseline.sh

# Check job status
squeue -u NETID

# View live output
tail -f logs/train_JOBID.out

# Cancel a job
scancel JOBID
```

## Resume after interruption

If a job times out or is cancelled, resume from last checkpoint:
```bash
# Add --resume to the python command in submit_train.sh, then resubmit
sbatch tempest/submit_train.sh
```
The `--resume` flag is already in `train_hpc.py` — just add it to the
`python` command in the SLURM script.

## Questions for UIT

- What partition should GPU jobs use?
- What are the correct `module load` names for Python and CUDA?
- Where should MIMIC-IV data be stored (scratch vs project storage)?
- What is the max wall time for GPU jobs?
- Is there a quota on GPU hours?
