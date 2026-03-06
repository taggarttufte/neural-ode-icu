"""
Neural ODE training on MIMIC-IV data.

Usage:
    python src/mimic_train_ode.py --epochs 1               # test run
    python src/mimic_train_ode.py --epochs 50              # full run
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_dataset import load_splits, collate_fn, N_VARS
from model import LatentODE

DATA_PATH = "/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz"
OUT_DIR   = "/home/j19w245/neural-ode-icu/results"


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            times   = batch['times'].to(device)
            values  = batch['values'].to(device)
            mask    = batch['mask'].to(device)
            seq_len = batch['seq_lengths'].to(device)
            probs   = model.predict_proba(times, values, mask, seq_len)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch['labels'].numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return roc_auc_score(labels, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch-size',  type=int,   default=64)
    parser.add_argument('--latent-dim',  type=int,   default=32)
    parser.add_argument('--hidden-dim',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = load_splits(DATA_PATH)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = LatentODE(
        input_dim=N_VARS,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_val_auroc = 0.0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            times   = batch['times'].to(device)
            values  = batch['values'].to(device)
            mask    = batch['mask'].to(device)
            seq_len = batch['seq_lengths'].to(device)
            labels  = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, kl_loss = model(times, values, mask, seq_len)
            probs = torch.sigmoid(logits)
            loss  = criterion(probs, labels) + 0.001 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_auroc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d} | loss {total_loss/len(train_loader):.4f} | val AUROC {val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{OUT_DIR}/mimic_ode_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    test_auroc = evaluate(model, test_loader, device)
    print(f"\nTest AUROC: {test_auroc:.4f}")
    print(f"Best Val AUROC: {best_val_auroc:.4f}")

    # Save results
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/mimic_ode_results.txt", "w") as f:
        f.write(f"epochs={args.epochs}\n")
        f.write(f"test_auroc={test_auroc:.4f}\n")
        f.write(f"best_val_auroc={best_val_auroc:.4f}\n")


if __name__ == "__main__":
    main()
