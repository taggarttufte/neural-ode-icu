#!/usr/bin/env python3
"""
mimic_eval_ode.py

Evaluates the trained Neural ODE model on the canonical test split.
Uses the actual LatentODE from model.py with predict_proba.

Requires:
  - data/canonical_split.npz
  - data/mimic_iv_processed_v2.npz
  - results/mimic_ode_best.pt

Saves: results/preds_ode.npz  {preds, labels}

Usage:
  python src/mimic_eval_ode.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_dataset import MIMICDataset, collate_fn, N_VARS
from model import LatentODE

STRUCTURED_PATH = "data/mimic_iv_processed_v2.npz"
SPLIT_PATH      = "data/canonical_split.npz"
MODEL_PATH      = "results/mimic_ode_best.pt"
SAVE_PATH       = "results/preds_ode.npz"

LATENT_DIM  = 32
HIDDEN_DIM  = 64
BATCH_SIZE  = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading canonical split...")
    split    = np.load(SPLIT_PATH)
    test_ids = set(split["test_ids"].tolist())

    print("Loading structured data...")
    s         = np.load(STRUCTURED_PATH)
    ts_ids    = s["stay_ids"].astype(int)
    ts_values = s["values"]    # (N, 48, 20)
    ts_masks  = s["masks"]     # (N, 48, 20)
    ts_times  = s["times"]     # (48,)
    ts_labels = s["labels"]

    # Select test patients
    mask_arr    = np.array([sid in test_ids for sid in ts_ids])
    test_values = ts_values[mask_arr]
    test_masks  = ts_masks[mask_arr]
    test_labels = ts_labels[mask_arr]
    print(f"Test patients : {test_values.shape[0]:,}  mortality : {test_labels.mean():.4f}")

    # Build dataset + loader (reuse existing classes)
    dataset = MIMICDataset(test_values, test_masks, ts_times, test_labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=0)

    print("Loading Neural ODE model...")
    model = LatentODE(N_VARS, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            times  = batch['times'].to(device)
            values = batch['values'].to(device)
            mask   = batch['mask'].to(device)
            seq_lengths = batch['seq_lengths']

            probs = model.predict_proba(times, values, mask, seq_lengths)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    print(f"Neural ODE  AUROC : {auroc:.4f}  AUPRC : {auprc:.4f}")

    np.savez(SAVE_PATH, preds=preds, labels=labels)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
