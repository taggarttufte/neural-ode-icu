"""Generate training curves and test set evaluation figures."""

import os, sys, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import PhysioNetDataset, collate_fn, N_VARS
from model import LatentODE
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ── 1. Load history ──────────────────────────────────────────────────────────
history = np.load(os.path.join(RESULTS_DIR, 'history.npy'), allow_pickle=True).item()
epochs = range(1, len(history['train_loss']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Neural ODE ICU Mortality — Training Curves', fontsize=13)

axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   color='tomato')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history['val_auroc'], label='Val AUROC', color='mediumseagreen')
axes[1].plot(epochs, history['val_auprc'], label='Val AUPRC', color='darkorange')
axes[1].axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Random baseline')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
axes[1].set_title('Validation Metrics'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.close()
print("Saved training_curves.png")

# ── 2. Load best model + run test inference ───────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(BASE_DIR, 'data', 'processed', 'test.pkl'), 'rb') as f:
    test_data = pickle.load(f)

test_loader = DataLoader(PhysioNetDataset(test_data), batch_size=64,
                         shuffle=False, collate_fn=collate_fn, num_workers=0)

model = LatentODE(input_dim=N_VARS, hidden_dim=64, latent_dim=32, ode_hidden_dim=64).to(device)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'checkpoints', 'best_model.pt'),
                                  map_location=device))
model.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        probs = model.predict_proba(
            batch['times'].to(device), batch['values'].to(device),
            batch['mask'].to(device),  batch['seq_lengths'].to(device)
        )
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

auroc = roc_auc_score(all_labels, all_probs)
auprc = average_precision_score(all_labels, all_probs)

# ── 3. ROC + PR curves ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Neural ODE ICU Mortality — Test Set Performance', fontsize=13)

# ROC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUROC = {auroc:.4f}')
axes[0].plot([0,1],[0,1], 'k--', lw=0.8, label='Random (0.50)')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve (Test Set)'); axes[0].legend(); axes[0].grid(alpha=0.3)

# PR
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
axes[1].plot(recall, precision, color='mediumseagreen', lw=2, label=f'AUPRC = {auprc:.4f}')
baseline = all_labels.mean()
axes[1].axhline(baseline, color='gray', linestyle='--', lw=0.8,
                label=f'Random ({baseline:.3f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve (Test Set)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'test_performance.png'), dpi=150)
plt.close()
print(f"Saved test_performance.png  |  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
