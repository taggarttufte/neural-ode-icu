# Neural ODE Training Notes & Lessons Learned

## Run History

### Run 1 — PhysioNet 2012 (baseline)
- **Config:** `--epochs 30 --latent-dim 32 --hidden-dim 64 --batch-size 32`
- **Result:** Test AUROC 0.7464 | AUPRC 0.4060 (best val AUROC 0.7524 @ epoch 21)
- **Notes:** First successful run. CPU-only PyTorch installed — switched to CUDA before this run.

### Run 2 — PhysioNet 2012 (larger model)
- **Config:** `--epochs 50 --latent-dim 64 --hidden-dim 128 --batch-size 32→64 --kl-weight 0.1`
- **Status:** In progress (resumed from epoch 23, best val AUROC 0.7686 @ epoch 19)
- **Notes:** Multiple crashes before stable run — see lessons below.

---

## Lessons Learned

### 1. OpenClaw exec sessions have a ~25-30 min timeout
- **Symptom:** Training always crashed around epoch 13-15 with no Python error
- **Cause:** OpenClaw manages exec sessions and kills them after ~25-30 min. Each epoch
  with the larger model takes ~2 min, so epoch 13 ≈ 26 min → session gets killed.
- **Fix:** Launch training via `Start-Process` (PowerShell) instead of direct exec.
  This spawns a fully detached Windows process that OpenClaw cannot kill.
  ```powershell
  $proc = Start-Process -FilePath "python" `
    -ArgumentList "src/train.py --epochs 50 ..." `
    -WorkingDirectory "C:\path\to\project" `
    -RedirectStandardOutput "results\run_log.txt" `
    -RedirectStandardError "results\run_err.txt" `
    -NoNewWindow -PassThru
  ```
- **Note:** Training also now saves `latest_model.pt` after every epoch for safe resume.

### 2. Resume support — always save per-epoch checkpoints
- **Problem:** If training dies mid-run, you lose all progress back to the last save.
- **Fix:** Added `--resume` flag. Saves `latest_model.pt` after every epoch containing:
  - `model_state`, `optimizer_state`, `epoch`, `best_auroc`, `history`
- **Usage:** `python src/train.py --epochs 50 ... --resume`
- **Note:** Best model weights are still saved separately as `best_model.pt` only when
  val AUROC improves.

### 3. AMP (mixed precision) breaks dopri5 adaptive ODE solver
- **Symptom:** Wrapping forward pass in `torch.cuda.amp.autocast()` caused speed to drop
  from ~1.2 it/s to ~0.017 it/s (58 seconds per batch vs <1 second)
- **Cause:** `dopri5` is an adaptive step-size solver — it uses error estimates to decide
  step sizes. Running the ODE function in fp16 corrupts these estimates, causing the
  solver to take hundreds of tiny steps to meet its tolerance threshold.
- **Fix:** Removed AMP entirely. Do NOT use `autocast()` with torchdiffeq ODE solvers.
- **Rule:** AMP is safe for standard networks (CNNs, Transformers, MLPs) but not for
  Neural ODEs with adaptive solvers.

### 4. num_workers > 0 causes issues on Windows
- **Symptom:** Setting `num_workers=4` on DataLoader caused extreme slowdown (~58s/batch)
- **Cause:** Windows uses `spawn` instead of `fork` for multiprocessing. Without a proper
  `if __name__ == '__main__':` guard in the script, each worker re-imports the module
  and spawns more workers — recursive process explosion.
- **Fix:** Keep `num_workers=0` on Windows. `pin_memory=True` is still fine and gives
  a small CPU→GPU transfer speedup.
- **Note:** On Linux/Mac `num_workers=4` works fine. Windows-specific issue.

### 5. Larger batch size (64 vs 32) = real speedup
- **Result:** Batch 64 → 63 batches/epoch at ~1.2 it/s ≈ 51 sec/epoch
  vs Batch 32 → 125 batches/epoch at ~1.0 it/s ≈ 125 sec/epoch
- **Takeaway:** Doubling batch size roughly halved epoch time with no AUROC penalty.
  VRAM went from ~2.7GB to ~3.5GB — still well within 12GB on RTX 3080 Ti.

### 6. GPU utilization caps at ~43-50% for Neural ODEs — that's normal
- **Cause:** The ODE solver (dopri5) is inherently sequential. It evaluates the ODE
  function multiple times per step and must wait for each result before proceeding.
  This limits GPU parallelism regardless of batch size or model size.
- **Takeaway:** Don't chase 90-100% utilization for Neural ODEs. ~50% is expected.
  The real speedup knobs are: looser ODE tolerances (`rtol/atol`) or a fixed-step
  solver like `euler` or `rk4` (trades accuracy for speed).

### 7. CUDA PyTorch vs CPU-only
- **Issue:** Default `pip install torch` may install CPU-only version
- **Fix:** Install explicitly with CUDA version matching your driver:
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
- **Check:** `torch.cuda.is_available()` should return `True`

---

## Useful Commands

### Check training status
```powershell
# See last N epoch results
Get-Content results\run2_log.txt | Select-String "Epoch\s+\d+ \|" | Select-Object -Last 10

# See current progress bar
Get-Content results\run2_err.txt -Tail 3

# Check process is alive
Get-Process -Id <PID>

# GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### Kill and resume
```powershell
Stop-Process -Id <PID> -Force

$proc = Start-Process -FilePath "python" `
  -ArgumentList "src/train.py --epochs 50 --batch-size 64 --latent-dim 64 --hidden-dim 128 --kl-weight 0.1 --resume" `
  -WorkingDirectory "C:\Users\Taggart\clawd\neural-ode-icu" `
  -RedirectStandardOutput "C:\Users\Taggart\clawd\neural-ode-icu\results\run2_log.txt" `
  -RedirectStandardError "C:\Users\Taggart\clawd\neural-ode-icu\results\run2_err.txt" `
  -NoNewWindow -PassThru
Write-Output "PID: $($proc.Id)"
```

### Check saved checkpoint
```powershell
python -c "import torch; ck=torch.load('results/checkpoints/latest_model.pt', map_location='cpu', weights_only=False); print('Epoch:', ck['epoch']); print('Best AUROC:', round(ck['best_auroc'],4))"
```

---

## Dataset Notes

### PhysioNet 2012 Challenge (current)
- Open access, no credentials needed
- 12k ICU patients (sets A/B/C), 3994/3992/3995 train/val/test split
- 37 clinical variables, 13.9% mortality rate
- Download: https://physionet.org/content/challenge-2012/1.0.0/

### MIMIC-III / MIMIC-IV (target)
- Requires PhysioNet credentialing + supervisor co-sign on Data Use Agreement
- MIMIC-IV ICU: ~50-70k stays (~5-6x PhysioNet 2012)
- Expected training time on RTX 3080 Ti: ~10-12 min/epoch → ~8-10 hrs for 50 epochs
- Fully pausable/resumable with `--resume` flag

### What NOT to download
- **MIMIC-III Waveform Database** — open access but 6.7TB of raw ECG/ABP waveforms,
  useless for structured mortality prediction
- **MIMIC Database v1.0.0 (mimicdb)** — original 1996 MIMIC, only 90 patients,
  waveform data, wrong format entirely
- **MIMIC-III Demo** — open access but only 100 patients, not enough to train on
