"""
Side-by-side comparison of all trained models on MIMIC-IV test set.

Generates:
  results/plots/compare_roc.png     - Overlay ROC curves for all available models
  results/plots/compare_summary.png - Bar chart of AUROC + AUPRC

Usage:
    python src/mimic_plot_compare.py
"""

import numpy as np
import os, re, glob
import torch
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_dataset import load_splits, collate_fn, N_VARS
from model import LatentODE
from clinicalbert.model import ClinicalBERTClassifier

DATA_PATH       = "/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz"
XGB_MODEL_PATH  = "/home/j19w245/neural-ode-icu/results/mimic_xgb_model.json"
ODE_MODEL_PATH  = "/home/j19w245/neural-ode-icu/results/mimic_ode_best.pt"
BERT_MODEL_PATH = "/home/j19w245/neural-ode-icu/results/mimic_bert_best.pt"
BERT_NAME       = "emilyalsentzer/Bio_ClinicalBERT"
OUT_DIR         = "/home/j19w245/neural-ode-icu/results/plots"

COLORS = {
    'XGBoost':    '#5b9af5',
    'Neural ODE': '#7ecfcf',
    'BERT':       '#f5a05b',
    'Random':     '#555566',
}

BERT_FEATURE_NAMES = [
    'HR', 'SBP', 'DBP', 'MBP', 'RespRate', 'Temp', 'SpO2',
    'Glucose', 'Creatinine', 'Sodium', 'Potassium', 'Hematocrit', 'WBC',
    'Bicarbonate', 'BUN', 'GCS_Motor', 'GCS_Verbal', 'GCS_Eye',
    'UrineOut', 'Age'
]


def patient_to_text(values, mask, max_chars=2000):
    parts = []
    for t in range(values.shape[0]):
        observed = [f"{BERT_FEATURE_NAMES[v]}={values[t, v]:.1f}"
                    for v in range(values.shape[1]) if mask[t, v] == 1]
        if observed:
            parts.append(f"Hr{t}: {', '.join(observed)}.")
    text = " ".join(parts)
    return text[:max_chars] if text else "No observations."


class MIMICTextDataset(Dataset):
    def __init__(self, values, masks, labels, tokenizer, max_length=512):
        self.values = values; self.masks = masks
        self.labels = labels; self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        text = patient_to_text(self.values[idx], self.masks[idx])
        enc  = self.tokenizer(text, max_length=self.max_length,
                              padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.float32)}

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


def extract_features(values, masks):
    N, T, V = values.shape
    features = np.zeros((N, V * 8), dtype=np.float32)
    for v in range(V):
        vals = values[:, :, v]; msk = masks[:, :, v]
        counts = msk.sum(axis=1); safe_counts = np.maximum(counts, 1)
        means = (vals * msk).sum(axis=1) / safe_counts
        stds  = np.sqrt(((vals - means[:, None]) ** 2 * msk).sum(axis=1) / safe_counts)
        mins  = np.where(msk == 1, vals, np.inf).min(axis=1);  mins[counts == 0] = 0.0
        maxs  = np.where(msk == 1, vals, -np.inf).max(axis=1); maxs[counts == 0] = 0.0
        times = np.arange(T, dtype=np.float32)
        first_vals = np.zeros(N); last_vals = np.zeros(N); trends = np.zeros(N)
        for i in range(N):
            obs_idx = np.where(msk[i] == 1)[0]
            if len(obs_idx) > 0:
                first_vals[i] = vals[i, obs_idx[0]]; last_vals[i] = vals[i, obs_idx[-1]]
                if len(obs_idx) > 1:
                    t_obs = times[obs_idx]; v_obs = vals[i, obs_idx]; t_c = t_obs - t_obs.mean()
                    if t_c.std() > 0:
                        trends[i] = np.dot(t_c, v_obs) / np.dot(t_c, t_c)
        base = v * 8
        features[:, base:base+8] = np.stack(
            [means, stds, mins, maxs, first_vals, last_vals, trends, counts], axis=1
        )
    return features


def get_xgb_probs(X_te, y_te):
    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)
    probs = model.predict_proba(X_te)[:, 1]
    return probs


def get_ode_probs(test_loader, device):
    model = LatentODE(input_dim=N_VARS, latent_dim=32, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(ODE_MODEL_PATH, map_location=device))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            probs = model.predict_proba(
                batch['times'].to(device),
                batch['values'].to(device),
                batch['mask'].to(device),
                batch['seq_lengths'].to(device),
            )
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch['labels'].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def get_bert_probs(values_te, masks_te, labels_te, device):
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    bert_ds   = MIMICTextDataset(values_te, masks_te, labels_te, tokenizer)
    loader    = DataLoader(bert_ds, batch_size=16, shuffle=False, num_workers=4)
    model = ClinicalBERTClassifier(BERT_NAME).to(device)
    model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            probs = torch.sigmoid(model(ids, attn)).cpu().numpy()
            all_probs.extend(probs.flatten())
    return np.array(all_probs), labels_te


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    print("Loading data...")
    d = np.load(DATA_PATH)
    X_all = extract_features(d['values'], d['masks'])
    y_all = d['labels']
    _, X_tmp, _, y_tmp = train_test_split(X_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
    _, X_te, _, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # For BERT we need raw values/masks on the same test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y_all))
    n_tr = int(len(y_all) * 0.70); n_va = int(len(y_all) * 0.15)
    te_idx = idx[n_tr + n_va:]
    values_te = d['values'][te_idx]
    masks_te  = d['masks'][te_idx]
    labels_te = d['labels'][te_idx]

    _, _, test_ds = load_splits(DATA_PATH)
    test_loader   = DataLoader(test_ds, batch_size=64, shuffle=False,
                               collate_fn=collate_fn, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}  # name -> (fpr, tpr, auroc, auprc)

    if os.path.exists(XGB_MODEL_PATH):
        print("Evaluating XGBoost...")
        xgb_probs = get_xgb_probs(X_te, y_te)
        fpr, tpr, _ = roc_curve(y_te, xgb_probs)
        auroc = roc_auc_score(y_te, xgb_probs)
        auprc = average_precision_score(y_te, xgb_probs)
        results['XGBoost'] = (fpr, tpr, auroc, auprc)
        print(f"  XGBoost  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    else:
        print(f"XGBoost model not found at {XGB_MODEL_PATH}")

    if os.path.exists(ODE_MODEL_PATH):
        print("Evaluating Neural ODE...")
        ode_probs, ode_labels = get_ode_probs(test_loader, device)
        fpr, tpr, _ = roc_curve(ode_labels, ode_probs)
        auroc = roc_auc_score(ode_labels, ode_probs)
        auprc = average_precision_score(ode_labels, ode_probs)
        results['Neural ODE'] = (fpr, tpr, auroc, auprc)
        print(f"  Neural ODE  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    else:
        print(f"ODE model not found at {ODE_MODEL_PATH}")

    if os.path.exists(BERT_MODEL_PATH):
        print("Evaluating ClinicalBERT (this may take a few minutes)...")
        bert_probs, bert_labels = get_bert_probs(values_te, masks_te, labels_te, device)
        fpr, tpr, _ = roc_curve(bert_labels, bert_probs)
        auroc = roc_auc_score(bert_labels, bert_probs)
        auprc = average_precision_score(bert_labels, bert_probs)
        results['BERT'] = (fpr, tpr, auroc, auprc)
        print(f"  BERT  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    else:
        print(f"BERT model not found at {BERT_MODEL_PATH}")

    if not results:
        print("No models found. Exiting.")
        return

    # --- Overlay ROC ---
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, (fpr, tpr, auroc, _) in results.items():
        c = COLORS.get(name, '#ffffff')
        ax.plot(fpr, tpr, color=c, lw=2.5, label=f'{name}  (AUROC = {auroc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.06, color=c)
    ax.plot([0, 1], [0, 1], color=COLORS['Random'], lw=1.2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate', labelpad=8)
    ax.set_ylabel('True Positive Rate', labelpad=8)
    ax.set_title('MIMIC-IV — ROC Curve Comparison', pad=14, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.3)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    roc_path = f"{OUT_DIR}/compare_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")

    # --- Bar Chart Summary ---
    names  = list(results.keys())
    aurocs = [results[n][2] for n in names]
    auprcs = [results[n][3] for n in names]
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width/2, aurocs, width, label='AUROC',
                   color=[COLORS.get(n, '#ffffff') for n in names], alpha=0.9, edgecolor='none')
    bars2 = ax.bar(x + width/2, auprcs, width, label='AUPRC',
                   color=[COLORS.get(n, '#ffffff') for n in names], alpha=0.5, edgecolor='none')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Score', labelpad=8)
    ax.set_title('MIMIC-IV — AUROC & AUPRC Summary', pad=14, fontsize=13, fontweight='bold')
    ax.legend(framealpha=0.3)
    ax.grid(True, axis='y', alpha=0.4)
    for bar, val in zip(list(bars1) + list(bars2), aurocs + auprcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, color='#e0e0e0')
    plt.tight_layout()
    bar_path = f"{OUT_DIR}/compare_summary.png"
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bar_path}")


if __name__ == "__main__":
    main()
