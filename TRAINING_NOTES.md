# Neural ODE Training Notes & Lessons Learned

## Run History

### Run 1 — PhysioNet 2012 (baseline)
- **Config:** `--epochs 30 --latent-dim 32 --hidden-dim 64 --batch-size 32`
- **Result:** Test AUROC 0.7464 | AUPRC 0.4060 (best val AUROC 0.7524 @ epoch 21)
- **Notes:** First successful run. CPU-only PyTorch installed — switched to CUDA before this run.

### Run 2 — PhysioNet 2012 (larger model)
- **Config:** `--epochs 50 --latent-dim 64 --hidden-dim 128 --batch-size 32→64 --kl-weight 0.1`
- **Result:** Test AUROC 0.7387 | AUPRC 0.3929 (best val AUROC 0.7686 @ epoch 19)
- **Notes:** Multiple crashes before stable run — see lessons below.

### Run 3 — PhysioNet 2012 (smaller model, early stopping)
- **Config:** `--epochs 50 --latent-dim 32 --hidden-dim 64 --batch-size 64 --early-stop-patience 10`
- **Result:** Test AUROC 0.7377 | AUPRC 0.3774 (early stopped @ epoch 28)
- **Notes:** Smaller model underperforms Run 2. Early stopping working as expected — training halted automatically when val AUROC plateaued.

---

## XGBoost Baseline

### XGBoost Run 1 — PhysioNet 2012 (hand-crafted features)
- **Features:** 367 per patient (36 vars x 10 statistical features + 2 global + 5 static)
  - Per variable: count, mean, std, min, max, first, last, trend, t_first, t_last
  - Global: total_obs, miss_rate
  - Static: Age, Gender, Height, ICUType, Weight
- **Config:** scale_pos_weight=6.21 (class imbalance), eval on val AUROC, early stopping
- **Result:** Test AUROC 0.8680 | Test AUPRC 0.5626 (Val AUROC 0.8697)
- **Training:** ~350 boosting rounds, converged quickly
- **Top features:** var10_last (GCS last value), var04_last (BUN last), var31_first (Temp first), var18_count (MechVent count), ICUType
- **Key insight:** Hand-crafted statistical features on tabular data dominate both Neural ODE
  and ClinicalBERT on this dataset size (12k patients). XGBoost benefits from explicit
  feature engineering that captures clinically meaningful patterns (last GCS, first Temp,
  ventilator frequency) without needing to learn them from raw time series.
- **Why it matters for the comparison:** Sets the ceiling for what's achievable on PhysioNet
  2012. If Neural ODE or ClinicalBERT can't beat this, the question is whether they
  close the gap on larger data (MIMIC-IV) where hand-crafted features may miss patterns
  that learned representations can capture.

---

## ClinicalBERT Runs

### BERT Run 1 — PhysioNet 2012 (normalized z-score values)
- **Model:** emilyalsentzer/Bio_ClinicalBERT (108.3M params, all fine-tuned)
- **Config:** `--epochs 5 --batch-size 32 --lr 2e-5 --fp16`
- **Serialization:** Patient time-series converted to text ("At hour X: HR=-0.8, Temp=0.4...")
  using z-score normalized values from preprocessed data
- **Result:** Test AUROC 0.7316 | Test AUPRC 0.2815 (Val AUROC 0.7227 @ epoch 5)
- **Notes:** Comparable to Neural ODE (~0.74). Both well below XGBoost (0.868).
  Still improving at epoch 5, may benefit from more epochs.

### BERT Run 2 — PhysioNet 2012 (denormalized real clinical values)
- **Model:** same as Run 1
- **Config:** same as Run 1
- **Serialization:** Denormalized values back to real clinical units before text conversion
  ("At hour X: HR=73.0, Temp=35.1, GCS=15.0...")
  using saved means.npy/stds.npy from preprocessing
- **Hypothesis:** ClinicalBERT was pretrained on real clinical notes with real values,
  so denormalized inputs should improve performance
- **Result:** Test AUROC 0.7001 | Test AUPRC 0.2417 (Val AUROC 0.6837 @ epoch 5)
- **Surprise:** Denormalized values performed WORSE than z-scores. Possible explanations:
  1. Z-scores center variables around 0 with unit variance, which may help the
     classifier head learn more easily
  2. Real clinical values have very different scales (HR~80, Temp~37, GCS~15, Urine~1000)
     which may cause gradient imbalance through the classifier
  3. BERT's positional and token embeddings may not meaningfully encode numerical magnitude
  4. The tokenizer fragments numbers unpredictably ("1000.0" becomes multiple tokens)
- **Next steps:** Try more epochs, try freezing BERT and only training the head,
  try adding variable-level normalization context to the text

### BERT Run 3 — 30 epochs, early stopping, z-score (normalized)
- **Config:** `--epochs 30 --batch-size 32 --lr 2e-5 --fp16 --early-stop-patience 5`
- **Serialization:** z-score normalized (same as Run 1)
- **Result:** Test AUROC 0.7256 | Test AUPRC 0.3004 (best val AUROC 0.7229 @ epoch 7)
- **Early stopped:** Epoch 12 (no improvement for 5 epochs)
- **Notes:** Clear overfitting after epoch 7 — train loss kept dropping (0.32→0.16)
  while val AUROC fell. More epochs are not the answer; bottleneck is representation.

### BERT Run 4 — Class weighting pos_weight=6.21
- **Config:** same as Run 3 + pos_weight=6.21 in BCEWithLogitsLoss
- **Result:** Test AUROC 0.7067 | Test AUPRC 0.2751 (best val AUROC 0.7027 @ epoch 5)
- **Early stopped:** Epoch 10
- **Notes:** Class weighting backfired — loss jumped from ~0.41 to ~1.20 because
  pos_weight=6.21 scales positive gradients 6x, effectively making LR too aggressive.
  Both AUROC and AUPRC got worse. Optimization instability, not a data problem.

### BERT Run 5 — Gentle class weighting pos_weight=2.0
- **Config:** same as Run 3 + pos_weight=2.0 in BCEWithLogitsLoss
- **Result:** Test AUROC 0.7161 | Test AUPRC 0.2747 (best val AUROC 0.7109 @ epoch 4)
- **Early stopped:** Epoch 9
- **Notes:** Loss well-behaved (0.62→0.42) but still underperforms unweighted runs.
  Class weighting consistently hurts regardless of strength. BERT peaks early (epoch 4-7)
  across all configs, suggesting a hard ceiling from the representation, not tuning.

### Key Finding: BERT ceiling on PhysioNet 2012
ClinicalBERT tops out at ~0.73 AUROC with serialized time-series. Consistent across
5 runs with different configs. Root cause: BERT treats numbers as tokens with no
numerical meaning — "73" and "82" are just different token sequences. Neural ODE
operates in continuous numerical space and explicitly models dynamics, making it
better suited to this data type.

Next step: compare using MIMIC-IV with actual clinical notes as BERT input (each
model gets its native input format) rather than serializing structured numbers.

### Comparison Table (PhysioNet 2012)
| Model                           | AUROC  | AUPRC  | Notes                    |
|---------------------------------|--------|--------|--------------------------|
| XGBoost (baseline)              | 0.8680 | 0.5626 | 367 engineered features  |
| Neural ODE Run 1                | 0.7464 | 0.4060 | best single run          |
| Neural ODE Run 2 (larger)       | 0.7387 | 0.3929 |                          |
| Neural ODE Run 3 (smaller)      | 0.7377 | 0.3774 | early stopped ep28       |
| ClinicalBERT Run 1 (5ep)        | 0.7316 | 0.2815 | best BERT AUROC          |
| ClinicalBERT Run 3 (30ep)       | 0.7256 | 0.3004 | best BERT AUPRC          |
| ClinicalBERT Run 5 (pw=2.0)     | 0.7161 | 0.2747 |                          |
| ClinicalBERT Run 4 (pw=6.21)    | 0.7067 | 0.2751 | unstable training        |
| ClinicalBERT Run 2 (denorm)     | 0.7001 | 0.2417 | worst — denorm hurts     |

---

## Lessons Learned

### 1. Shell/exec session timeouts kill long training runs
- **Symptom:** Training always crashed around epoch 13-15 with no Python error
- **Cause:** Interactive shell sessions (terminal, VS Code integrated terminal, etc.) have
  implicit timeouts or get killed when the window closes. Each epoch with the larger model
  takes ~2 min, so a 30-min session dies around epoch 13-15 with no Python error.
- **Fix:** Launch training via `Start-Process` (PowerShell) instead of direct exec.
  This spawns a fully detached Windows process that survives session termination.
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

### 5. Overfitting — add early stopping
- **Symptom:** Run 2 peaked at AUROC 0.7686 (epoch 19) then declined to 0.7324 by
  epoch 37 while train loss kept dropping — classic overfitting.
- **Fix:** Added `--early-stop-patience N` (default 10). Stops training if val AUROC
  hasn't improved in N consecutive epochs. Run 2 would have stopped at epoch 29
  instead of wasting ~20 epochs past peak.
- **Usage:** `python src/train.py ... --early-stop-patience 10`
  Set to 0 to disable.

### 6. Larger batch size (64 vs 32) = real speedup
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

## MIMIC-IV Runs (2026-03-05)

### Data
- mimic_iv_processed_v2.npz: 74829 patients, 48h, 20 features (18 chartevents + urine_output + age)
- Mortality: 11.9%
- Split: 70/15/15

### XGBoost (COMPLETE)
- Test AUROC: 0.9106 | AUPRC: 0.6652
- Val AUROC: 0.9066 | AUPRC: 0.6571
- 160 features (20 vars x 8 stats: mean/std/min/max/first/last/trend/count)

### Neural ODE (1-epoch test)
- Test AUROC: 0.8581 | Val: 0.8645
- Full 50-epoch run: RUNNING overnight

### ClinicalBERT
- Blocked by PyTorch 2.5 / transformers torch.load CVE � upgraded to 2.6
- Also needs: pip install accelerate sentencepiece protobuf
- Status: 1-epoch test pending
