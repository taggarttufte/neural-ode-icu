"""
Generate result figures:
  1. Run 3 training curves (loss + AUROC/AUPRC over epochs)
  2. Run 2 vs Run 3 AUROC comparison
  3. Run 3 test set ROC + Precision-Recall curves
"""

import os
import re
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, roc_auc_score)
import torch
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from dataset import PhysioNetDataset, collate_fn, N_VARS
from model import LatentODE
from torch.utils.data import DataLoader


def parse_log(path):
    """Parse epoch metrics from a training log file."""
    epochs, train_loss, val_loss, auroc, auprc = [], [], [], [], []
    pattern = re.compile(
        r'Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)'
        r'\s+\|\s+AUROC:\s+([\d.]+)\s+\|\s+AUPRC:\s+([\d.]+)'
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                auroc.append(float(m.group(4)))
                auprc.append(float(m.group(5)))
    return (np.array(epochs), np.array(train_loss),
            np.array(val_loss), np.array(auroc), np.array(auprc))


def main():
    results_dir = os.path.join(BASE_DIR, 'results')
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')

    # ── Parse logs ──────────────────────────────────────────────────────────
    run2_log = os.path.join(results_dir, 'run2_log.txt')
    run3_log = os.path.join(results_dir, 'run3_log.txt')

    ep2, tl2, vl2, au2, ap2 = parse_log(run2_log)
    ep3, tl3, vl3, au3, ap3 = parse_log(run3_log)
    print(f"Run 2: {len(ep2)} epochs parsed")
    print(f"Run 3: {len(ep3)} epochs parsed (early stopped)")

    # ── Figure 1: Run 3 training curves ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Run 3 — Training Curves\n'
                 '(latent-dim 32, hidden-dim 64, batch 64, early stop @ epoch 28)',
                 fontsize=13, fontweight='bold')

    # Loss
    axes[0].plot(ep3, tl3, label='Train Loss', color='steelblue', lw=2)
    axes[0].plot(ep3, vl3, label='Val Loss',   color='coral',     lw=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    # AUROC / AUPRC
    axes[1].plot(ep3, au3, label='Val AUROC', color='steelblue', lw=2)
    axes[1].plot(ep3, ap3, label='Val AUPRC', color='coral',     lw=2)
    best_ep = ep3[np.argmax(au3)]
    best_au = au3.max()
    axes[1].axvline(best_ep, color='gray', linestyle='--', lw=1.2,
                    label=f'Best AUROC={best_au:.4f} (ep {best_ep})')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
    axes[1].set_title('Validation AUROC / AUPRC')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(results_dir, 'run3_training_curves.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")

    # ── Figure 2: Run 2 vs Run 3 AUROC comparison ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Run 2 vs Run 3 — Validation AUROC & AUPRC',
                 fontsize=13, fontweight='bold')

    for ax, metric2, metric3, ylabel in [
        (axes[0], au2, au3, 'Val AUROC'),
        (axes[1], ap2, ap3, 'Val AUPRC'),
    ]:
        ax.plot(ep2, metric2, label='Run 2 (latent-dim 64)', color='coral',     lw=2)
        ax.plot(ep3, metric3, label='Run 3 (latent-dim 32)', color='steelblue', lw=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(results_dir, 'run2_vs_run3.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")

    # ── Figure 3: Run 3 test set ROC + PR ────────────────────────────────────
    print("Running test set inference with Run 3 best model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = LatentODE(input_dim=N_VARS, hidden_dim=64,
                        latent_dim=32, ode_hidden_dim=64).to(device)
    model.load_state_dict(torch.load(
        os.path.join(results_dir, 'checkpoints', 'best_model.pt'),
        map_location=device, weights_only=True))
    model.eval()

    with open(os.path.join(processed_dir, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)

    loader = DataLoader(PhysioNetDataset(test), batch_size=64,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            p = model.predict_proba(
                batch['times'].to(device), batch['values'].to(device),
                batch['mask'].to(device), batch['seq_lengths'].to(device))
            all_probs.extend(p.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    test_auroc = roc_auc_score(labels, probs)
    test_auprc = average_precision_score(labels, probs)
    print(f"Run 3 Test AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Run 3 — Test Set Performance  '
                 f'(AUROC={test_auroc:.4f}, AUPRC={test_auprc:.4f})',
                 fontsize=13, fontweight='bold')

    fpr, tpr, _ = roc_curve(labels, probs)
    axes[0].plot(fpr, tpr, color='steelblue', lw=2,
                 label=f'Neural ODE (AUROC={test_auroc:.4f})')
    axes[0].plot([0,1],[0,1],'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(labels, probs)
    axes[1].plot(rec, prec, color='steelblue', lw=2,
                 label=f'Neural ODE (AUPRC={test_auprc:.4f})')
    axes[1].axhline(labels.mean(), color='k', linestyle='--', lw=1,
                    label=f'Baseline ({labels.mean():.3f})')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out3 = os.path.join(results_dir, 'run3_test_performance.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out3}")


if __name__ == '__main__':
    main()
