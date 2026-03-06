"""
Neural ODE plots for MIMIC-IV results.

Generates:
  results/plots/ode_training_curve.png  - Val AUROC per epoch (parsed from SLURM log)
  results/plots/ode_roc.png             - ROC curve on test set

Usage:
    python src/mimic_plot_ode.py [--log PATH_TO_SLURM_LOG]

If --log is not given, searches for slurm-*.out in common locations.
"""

import argparse
import re
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_dataset import load_splits, collate_fn, N_VARS
from model import LatentODE

DATA_PATH  = "/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz"
MODEL_PATH = "/home/j19w245/neural-ode-icu/results/mimic_ode_best.pt"
OUT_DIR    = "/home/j19w245/neural-ode-icu/results/plots"

STYLE = {
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#3a3d4d',
    'axes.labelcolor':  '#e0e0e0',
    'text.color':       '#e0e0e0',
    'xtick.color':      '#a0a0b0',
    'ytick.color':      '#a0a0b0',
    'grid.color':       '#2a2d3d',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
}


def find_ode_log():
    """Search likely locations for the ODE SLURM log file."""
    search_dirs = [
        "/home/j19w245/neural-ode-icu/src",
        "/home/j19w245/neural-ode-icu",
        "/home/j19w245",
    ]
    candidates = []
    for d in search_dirs:
        candidates.extend(glob.glob(f"{d}/slurm-*.out"))
        candidates.extend(glob.glob(f"{d}/ode*.log"))
        candidates.extend(glob.glob(f"{d}/mimic_ode*.log"))

    # Filter to ones that contain "val AUROC" (ODE training output)
    ode_logs = []
    for c in candidates:
        try:
            with open(c) as f:
                content = f.read()
            if 'val AUROC' in content or 'val auroc' in content.lower():
                ode_logs.append((os.path.getmtime(c), c))
        except Exception:
            pass

    if not ode_logs:
        return None
    # Return most recently modified
    ode_logs.sort(reverse=True)
    return ode_logs[0][1]


def parse_training_log(log_path):
    """
    Parse lines like:
      Epoch   1 | loss 0.3412 | val AUROC 0.8012
    Returns (epochs, losses, val_aurocs).
    """
    epoch_pat = re.compile(
        r'Epoch\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*val AUROC\s+([\d.]+)',
        re.IGNORECASE
    )
    epochs, losses, aurocs = [], [], []
    with open(log_path) as f:
        for line in f:
            m = epoch_pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                aurocs.append(float(m.group(3)))
    return epochs, losses, aurocs


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
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',        type=str, default=None,
                        help='Path to SLURM log with training output')
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    # --- Training Curve ---
    log_path = args.log or find_ode_log()
    if log_path and os.path.exists(log_path):
        print(f"Parsing log: {log_path}")
        epochs, losses, aurocs = parse_training_log(log_path)
        if epochs:
            best_epoch = epochs[aurocs.index(max(aurocs))]
            best_auroc = max(aurocs)
            print(f"Best val AUROC: {best_auroc:.4f} at epoch {best_epoch}")

            fig, ax1 = plt.subplots(figsize=(8, 5))
            color_auroc = '#5b9af5'
            color_loss  = '#f5a05b'

            ax1.plot(epochs, aurocs, color=color_auroc, lw=2.5, marker='o',
                     markersize=4, label='Val AUROC')
            ax1.axvline(best_epoch, color=color_auroc, lw=1.2, linestyle='--', alpha=0.5)
            ax1.scatter([best_epoch], [best_auroc], color='white', s=60, zorder=5,
                        edgecolors=color_auroc, linewidths=1.5)
            ax1.set_xlabel('Epoch', labelpad=8)
            ax1.set_ylabel('Val AUROC', color=color_auroc, labelpad=8)
            ax1.tick_params(axis='y', labelcolor=color_auroc)
            ax1.set_ylim(bottom=max(0, min(aurocs) - 0.02))
            ax1.grid(True, alpha=0.35)
            ax1.annotate(f'Best: {best_auroc:.4f} (ep {best_epoch})',
                         xy=(best_epoch, best_auroc),
                         xytext=(best_epoch + max(1, len(epochs)*0.05), best_auroc - 0.008),
                         color='#e0e0e0', fontsize=9, alpha=0.85)

            ax2 = ax1.twinx()
            ax2.set_facecolor('#1a1d27')
            ax2.plot(epochs, losses, color=color_loss, lw=1.8, linestyle='--',
                     alpha=0.7, label='Train Loss')
            ax2.set_ylabel('Train Loss', color=color_loss, labelpad=8)
            ax2.tick_params(axis='y', labelcolor=color_loss)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc='lower right', framealpha=0.3)

            fig.suptitle('Neural ODE — Training Curve (MIMIC-IV)', fontsize=13,
                         fontweight='bold', y=1.01)
            plt.tight_layout()
            curve_path = f"{OUT_DIR}/ode_training_curve.png"
            plt.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {curve_path}")
        else:
            print("Warning: no epoch lines found in log — skipping training curve")
    else:
        print(f"No ODE log found (pass --log). Skipping training curve.")

    # --- ROC Curve ---
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Skipping ROC curve.")
        return

    print("Loading data for test set...")
    _, _, test_ds = load_splits(DATA_PATH)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = LatentODE(
        input_dim=N_VARS,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    probs, labels = evaluate(model, test_loader, device)
    auroc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    print(f"Test AUROC: {auroc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='#7ecfcf', lw=2.5, label=f'Neural ODE (AUROC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='#555566', lw=1.2, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#7ecfcf')
    ax.set_xlabel('False Positive Rate', labelpad=8)
    ax.set_ylabel('True Positive Rate', labelpad=8)
    ax.set_title('Neural ODE — ROC Curve (MIMIC-IV Test Set)', pad=14,
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.3)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    roc_path = f"{OUT_DIR}/ode_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")


if __name__ == "__main__":
    main()
