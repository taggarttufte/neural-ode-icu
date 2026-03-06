"""
Generate result plots for the XGBoost baseline and a combined comparison figure.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, roc_auc_score)
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_VARS = 36


def extract_features(patient):
    times  = np.array(patient['times'])
    values = np.array(patient['values'])
    mask   = np.array(patient['mask'])
    static = patient['static']
    features = []
    for v in range(N_VARS):
        obs_mask  = mask[:, v].astype(bool)
        obs_vals  = values[obs_mask, v]
        obs_times = times[obs_mask]
        count = obs_mask.sum()
        if count == 0:
            features.extend([0]*10)
        else:
            mean_v  = obs_vals.mean()
            std_v   = obs_vals.std() if count > 1 else 0.0
            min_v   = obs_vals.min()
            max_v   = obs_vals.max()
            first_v = obs_vals[0]
            last_v  = obs_vals[-1]
            t_first = obs_times[0]
            t_last  = obs_times[-1]
            if count > 1:
                t_c = obs_times - obs_times.mean()
                denom = (t_c**2).sum()
                trend = (t_c * obs_vals).sum() / denom if denom > 0 else 0.0
            else:
                trend = 0.0
            features.extend([count, mean_v, std_v, min_v, max_v,
                              first_v, last_v, trend, t_first, t_last])
    total_obs = mask.sum()
    miss_rate = 1.0 - mask.mean()
    features.extend([total_obs, miss_rate])
    for key in ['Age', 'Gender', 'Height', 'ICUType', 'Weight']:
        val = static.get(key, -1.0)
        features.append(val if val is not None else -1.0)
    return np.array(features, dtype=np.float32)


def main():
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    results_dir   = os.path.join(BASE_DIR, 'results')

    print("Loading data and model...")
    with open(os.path.join(processed_dir, 'train.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(processed_dir, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)

    X_train = np.nan_to_num(np.stack([extract_features(p) for p in train]))
    X_test  = np.nan_to_num(np.stack([extract_features(p) for p in test]))
    y_test  = np.array([p['label'] for p in test])

    scaler  = StandardScaler().fit(X_train)
    X_test  = scaler.transform(X_test)

    model = xgb.XGBClassifier()
    model.load_model(os.path.join(results_dir, 'xgboost_baseline.json'))
    xgb_probs = model.predict_proba(X_test)[:, 1]

    # ── Figure 1: XGBoost ROC + PR ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('XGBoost Baseline — Test Set Performance', fontsize=14, fontweight='bold')

    # ROC
    fpr, tpr, _ = roc_curve(y_test, xgb_probs)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'XGBoost (AUROC = {roc_auc:.4f})')
    axes[0].plot([0,1],[0,1],'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

    # PR
    prec, rec, _ = precision_recall_curve(y_test, xgb_probs)
    ap = average_precision_score(y_test, xgb_probs)
    axes[1].plot(rec, prec, color='darkorange', lw=2,
                 label=f'XGBoost (AUPRC = {ap:.4f})')
    axes[1].axhline(y=y_test.mean(), color='k', linestyle='--', lw=1, label='Baseline (prevalence)')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(results_dir, 'baseline_performance.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")

    # ── Figure 2: Combined ROC comparison ───────────────────────────────────
    # Load Neural ODE run 1 best model predictions
    try:
        import torch
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
        from dataset import PhysioNetDataset, collate_fn
        from model import LatentODE
        from torch.utils.data import DataLoader

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_ode = LatentODE(input_dim=N_VARS, hidden_dim=64, latent_dim=32, ode_hidden_dim=64).to(device)
        model_ode.load_state_dict(torch.load(
            os.path.join(results_dir, 'checkpoints', 'best_model.pt'),
            map_location=device, weights_only=True))
        model_ode.eval()

        test_loader = DataLoader(PhysioNetDataset(test), batch_size=64,
                                  shuffle=False, collate_fn=collate_fn, num_workers=0)
        ode_probs, ode_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                p = model_ode.predict_proba(
                    batch['times'].to(device), batch['values'].to(device),
                    batch['mask'].to(device), batch['seq_lengths'].to(device))
                ode_probs.extend(p.cpu().numpy())
                ode_labels.extend(batch['labels'].numpy())
        ode_probs  = np.array(ode_probs)
        ode_labels = np.array(ode_labels)

        fig, ax = plt.subplots(figsize=(7, 6))
        for probs, labels, name, color in [
            (xgb_probs, y_test,    'XGBoost (AUROC=0.868)', 'darkorange'),
            (ode_probs, ode_labels, 'Neural ODE (AUROC=0.746)', 'steelblue'),
        ]:
            fpr, tpr, _ = roc_curve(labels, probs)
            ax.plot(fpr, tpr, color=color, lw=2, label=name)
        ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison — XGBoost vs Neural ODE', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(results_dir, 'model_comparison.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out2}")
    except Exception as e:
        print(f"Skipped combined comparison plot: {e}")


if __name__ == '__main__':
    main()
