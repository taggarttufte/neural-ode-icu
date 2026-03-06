"""
Load best saved ODE checkpoint and evaluate on test set.
Run: python src/mimic_eval_ode.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_dataset import load_splits, collate_fn, N_VARS
from model import LatentODE

DATA_PATH  = "/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz"
CKPT_PATH  = "/home/j19w245/neural-ode-icu/results/mimic_ode_best.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, _, test_ds = load_splits(DATA_PATH)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

model = LatentODE(input_dim=N_VARS, latent_dim=32, hidden_dim=64).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

probs, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        p = model.predict_proba(batch['times'].to(device), batch['values'].to(device),
                                batch['mask'].to(device), batch['seq_lengths'].to(device))
        probs.append(p.cpu().numpy())
        labels.append(batch['labels'].numpy())

probs  = np.concatenate(probs)
labels = np.concatenate(labels)
auroc  = roc_auc_score(labels, probs)
auprc  = average_precision_score(labels, probs)
print(f"Best checkpoint (epoch 16)")
print(f"Test AUROC: {auroc:.4f}")
print(f"Test AUPRC: {auprc:.4f}")

with open("/home/j19w245/neural-ode-icu/results/mimic_ode_results.txt", "w") as f:
    f.write(f"best_epoch=16\ntest_auroc={auroc:.4f}\ntest_auprc={auprc:.4f}\n")
