"""
XGBoost baseline on MIMIC-IV data.

Extracts per-variable statistical features from the time series,
then trains XGBoost. Same strategy as PhysioNet 2012 baseline.

Features per variable (20 vars x 8 = 160):
  mean, std, min, max, first, last, trend, count

Usage:
    python src/mimic_train_xgb.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

DATA_PATH = "/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz"
OUT_DIR   = "/home/j19w245/neural-ode-icu/results"

FEATURE_NAMES = [
    'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temp_c', 'spo2',
    'glucose', 'creatinine', 'sodium', 'potassium', 'hematocrit', 'wbc',
    'bicarbonate', 'bun', 'gcs_motor', 'gcs_verbal', 'gcs_eye',
    'urine_output', 'age'
]


def extract_features(values, masks):
    """
    values : (N, 48, 20)
    masks  : (N, 48, 20)
    returns: (N, 160) feature matrix
    """
    N, T, V = values.shape
    features = np.zeros((N, V * 8), dtype=np.float32)

    for v in range(V):
        vals = values[:, :, v]   # (N, 48)
        msk  = masks[:, :, v]    # (N, 48)

        # Count observations per patient
        counts = msk.sum(axis=1)  # (N,)

        # Masked mean
        safe_counts = np.maximum(counts, 1)
        means = (vals * msk).sum(axis=1) / safe_counts

        # Masked std
        sq_diff = ((vals - means[:, None]) ** 2) * msk
        stds = np.sqrt(sq_diff.sum(axis=1) / safe_counts)

        # Min / max (where observed)
        vals_masked = np.where(msk == 1, vals, np.inf)
        mins = vals_masked.min(axis=1)
        mins[counts == 0] = 0.0

        vals_masked2 = np.where(msk == 1, vals, -np.inf)
        maxs = vals_masked2.max(axis=1)
        maxs[counts == 0] = 0.0

        # First / last observed value
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
                    t_obs = times[obs_idx]
                    v_obs = vals[i, obs_idx]
                    t_c   = t_obs - t_obs.mean()
                    if t_c.std() > 0:
                        trends[i] = np.dot(t_c, v_obs) / np.dot(t_c, t_c)

        base = v * 8
        features[:, base + 0] = means
        features[:, base + 1] = stds
        features[:, base + 2] = mins
        features[:, base + 3] = maxs
        features[:, base + 4] = first_vals
        features[:, base + 5] = last_vals
        features[:, base + 6] = trends
        features[:, base + 7] = counts

    return features


def main():
    print("Loading data...")
    d = np.load(DATA_PATH)
    values = d['values']
    masks  = d['masks']
    labels = d['labels']

    print("Extracting features...")
    X = extract_features(values, masks)
    y = labels
    print(f"Feature matrix: {X.shape}")

    # Split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_va, X_te, y_va, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
    print(f"Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=20,
        device='cuda',
        random_state=42,
        n_jobs=4,
    )

    print("Training XGBoost...")
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

    val_auroc  = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    test_auroc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    val_auprc  = average_precision_score(y_va, model.predict_proba(X_va)[:, 1])
    test_auprc = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])

    print(f"\nVal  AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_model(f"{OUT_DIR}/mimic_xgb_model.json")
    with open(f"{OUT_DIR}/mimic_xgb_results.txt", "w") as f:
        f.write(f"test_auroc={test_auroc:.4f}\n")
        f.write(f"test_auprc={test_auprc:.4f}\n")
        f.write(f"val_auroc={val_auroc:.4f}\n")
        f.write(f"val_auprc={val_auprc:.4f}\n")
    print(f"Saved to {OUT_DIR}/mimic_xgb_results.txt")


if __name__ == "__main__":
    main()
