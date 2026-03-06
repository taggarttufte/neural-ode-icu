"""
XGBoost plots for MIMIC-IV results.

Generates:
  results/plots/xgb_roc.png            - ROC curve on test set
  results/plots/xgb_feature_importance.png - Top 20 features by gain

Usage:
    python src/mimic_plot_xgb.py
"""

import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import os

DATA_PATH  = "/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz"
MODEL_PATH = "/home/j19w245/neural-ode-icu/results/mimic_xgb_model.json"
OUT_DIR    = "/home/j19w245/neural-ode-icu/results/plots"

FEATURE_NAMES = [
    'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temp_c', 'spo2',
    'glucose', 'creatinine', 'sodium', 'potassium', 'hematocrit', 'wbc',
    'bicarbonate', 'bun', 'gcs_motor', 'gcs_verbal', 'gcs_eye',
    'urine_output', 'age'
]
STAT_NAMES = ['mean', 'std', 'min', 'max', 'first', 'last', 'trend', 'count']
ALL_FEATURE_NAMES = [f"{v}_{s}" for v in FEATURE_NAMES for s in STAT_NAMES]

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
        vals = values[:, :, v]
        msk  = masks[:, :, v]
        counts = msk.sum(axis=1)
        safe_counts = np.maximum(counts, 1)
        means = (vals * msk).sum(axis=1) / safe_counts
        sq_diff = ((vals - means[:, None]) ** 2) * msk
        stds = np.sqrt(sq_diff.sum(axis=1) / safe_counts)
        vals_inf = np.where(msk == 1, vals, np.inf)
        mins = vals_inf.min(axis=1); mins[counts == 0] = 0.0
        vals_ninf = np.where(msk == 1, vals, -np.inf)
        maxs = vals_ninf.max(axis=1); maxs[counts == 0] = 0.0
        times = np.arange(T, dtype=np.float32)
        first_vals = np.zeros(N, dtype=np.float32)
        last_vals  = np.zeros(N, dtype=np.float32)
        trends     = np.zeros(N, dtype=np.float32)
        for i in range(N):
            obs_idx = np.where(msk[i] == 1)[0]
            if len(obs_idx) > 0:
                first_vals[i] = vals[i, obs_idx[0]]
                last_vals[i]  = vals[i, obs_idx[-1]]
                if len(obs_idx) > 1:
                    t_obs = times[obs_idx]; v_obs = vals[i, obs_idx]
                    t_c = t_obs - t_obs.mean()
                    if t_c.std() > 0:
                        trends[i] = np.dot(t_c, v_obs) / np.dot(t_c, t_c)
        base = v * 8
        features[:, base:base+8] = np.stack(
            [means, stds, mins, maxs, first_vals, last_vals, trends, counts], axis=1
        )
    return features


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update(STYLE)

    print("Loading data...")
    d = np.load(DATA_PATH)
    X = extract_features(d['values'], d['masks'])
    y = d['labels']

    _, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    _, X_te, _, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    print("Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    probs = model.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, probs)
    fpr, tpr, _ = roc_curve(y_te, probs)
    print(f"Test AUROC: {auroc:.4f}")

    # --- ROC Curve ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='#5b9af5', lw=2.5, label=f'XGBoost (AUROC = {auroc:.4f})')
    ax.plot([0, 1], [0, 1], color='#555566', lw=1.2, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#5b9af5')
    ax.set_xlabel('False Positive Rate', labelpad=8)
    ax.set_ylabel('True Positive Rate', labelpad=8)
    ax.set_title('XGBoost — ROC Curve (MIMIC-IV Test Set)', pad=14, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.3)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    roc_path = f"{OUT_DIR}/xgb_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")

    # --- Feature Importance (Top 20 by gain) ---
    importance = model.get_booster().get_score(importance_type='gain')
    # Map XGBoost feature names (f0, f1, ...) to readable names
    named = {ALL_FEATURE_NAMES[int(k[1:])]: v for k, v in importance.items()
             if k[1:].isdigit() and int(k[1:]) < len(ALL_FEATURE_NAMES)}
    sorted_feats = sorted(named.items(), key=lambda x: x[1], reverse=True)[:20]
    feat_labels  = [f[0] for f in reversed(sorted_feats)]
    feat_values  = [f[1] for f in reversed(sorted_feats)]

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#5b9af5' if 'gcs' in l or 'age' in l or 'urine' in l else '#7ecfcf'
              for l in feat_labels]
    bars = ax.barh(feat_labels, feat_values, color=colors, height=0.65, edgecolor='none')
    ax.set_xlabel('Gain (mean improvement per split)', labelpad=8)
    ax.set_title('XGBoost — Top 20 Feature Importances (MIMIC-IV)', pad=14,
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.4)
    # Subtle value labels
    for bar, val in zip(bars, feat_values):
        ax.text(val + max(feat_values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', ha='left', fontsize=8.5, color='#a0a0b0')
    plt.tight_layout()
    imp_path = f"{OUT_DIR}/xgb_feature_importance.png"
    plt.savefig(imp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {imp_path}")


if __name__ == "__main__":
    main()
