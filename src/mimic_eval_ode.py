#!/usr/bin/env python3
"""
mimic_eval_ode.py

Evaluates the trained Neural ODE model on the canonical test split.
Saves per-patient predictions for bootstrap CI analysis.

Requires:
  - data/canonical_split.npz  (from mimic_create_split.py)
  - data/mimic_iv_processed_v2.npz
  - results/mimic_ode_best.pt

Saves: results/preds_ode.npz  {preds, labels}

Usage:
  python src/mimic_eval_ode.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from sklearn.metrics import roc_auc_score, average_precision_score
import os

STRUCTURED_PATH = "data/mimic_iv_processed_v2.npz"
SPLIT_PATH      = "data/canonical_split.npz"
MODEL_PATH      = "results/mimic_ode_best.pt"
SAVE_PATH       = "results/preds_ode.npz"

LATENT_DIM  = 32
HIDDEN_DIM  = 64
INPUT_DIM   = 20
BATCH_SIZE  = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Model (must match mimic_train_ode.py exactly) ─────────────────────────────
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, t, z):
        return self.net(z)


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.gru  = nn.GRU(input_dim * 2, hidden_dim, batch_first=True)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=-1)
        inp_rev = torch.flip(inp, dims=[1])
        _, h = self.gru(inp_rev)
        h = h.squeeze(0)
        return self.mean(h), self.logvar(h)


class LatentODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim)
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.decoder  = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.t_span = torch.linspace(0, 1, 2)

    def forward(self, x, mask):
        mean, logvar = self.encoder(x, mask)
        z0 = mean   # deterministic at inference
        t_span = self.t_span.to(z0.device)
        zt = odeint(self.ode_func, z0, t_span, method="dopri5")
        z_final = zt[-1]
        return self.decoder(z_final).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class MIMICTestDataset(Dataset):
    def __init__(self, data, labels):
        self.data   = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x     = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        mask  = (x != 0).float()
        return x, mask, label


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading canonical split...")
    split    = np.load(SPLIT_PATH)
    test_ids = set(split["test_ids"].tolist())

    print("Loading structured data...")
    s        = np.load(STRUCTURED_PATH)
    ts_ids   = s["stay_ids"].astype(int)
    ts_data  = s["data"]
    ts_labels = s["labels"].astype(int)

    mask_arr    = np.array([sid in test_ids for sid in ts_ids])
    test_data   = ts_data[mask_arr]
    test_labels = ts_labels[mask_arr]
    print(f"Test patients : {test_data.shape[0]:,}  mortality : {test_labels.mean():.4f}")

    dataset = MIMICTestDataset(test_data, test_labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    print("Loading Neural ODE model...")
    model = LatentODE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, mask, labels in loader:
            x, mask = x.to(device), mask.to(device)
            logits = model(x, mask)
            preds  = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    print(f"Neural ODE  AUROC : {auroc:.4f}  AUPRC : {auprc:.4f}")

    np.savez(SAVE_PATH, preds=preds, labels=labels)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
