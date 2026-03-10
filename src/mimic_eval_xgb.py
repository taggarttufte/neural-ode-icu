#!/usr/bin/env python3
"""
mimic_eval_xgb.py

Evaluates the trained XGBoost model on the canonical test split.
Saves per-patient predictions for bootstrap CI analysis.

Requires:
  - data/canonical_split.npz  (from mimic_create_split.py)
  - data/mimic_iv_processed_v2.npz
  - results/mimic_xgb_model.json

Saves: results/preds_xgb.npz  {preds, labels}

Usage:
  python src/mimic_eval_xgb.py
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import os

STRUCTURED_PATH = "data/mimic_iv_processed_v2.npz"
SPLIT_PATH      = "data/canonical_split.npz"
MODEL_PATH      = "results/mimic_xgb_model.json"
SAVE_PATH       = "results/preds_xgb.npz"


def extract_features(values, masks):
    """
    values : (N, 48, 20)
    masks  : (N, 48, 20)
    returns: (N, 160) feature matrix — 8 features per variable
    Must match mimic_train_xgb.py exactly.
    """
    N, T, V = values.shape
    features = np.zeros((N, V * 8), dtype=np.float32)
    times = np.arange(T, dtype=np.float32)

    for v in range(V):
        vals = values[:, :, v]
        msk  = masks[:, :, v]
        counts = msk.sum(axis=1)
        safe_counts = np.maximum(counts, 1)
        means = (vals * msk).sum(axis=1) / safe_counts
        sq_diff = ((vals - means[:, None]) ** 2) * msk
        stds = np.sqrt(sq_diff.sum(axis=1) / safe_counts)

        vals_masked = np.where(msk == 1, vals, np.inf)
        mins = vals_masked.min(axis=1)
        mins[counts == 0] = 0.0
        vals_masked2 = np.where(msk == 1, vals, -np.inf)
        maxs = vals_masked2.max(axis=1)
        maxs[counts == 0] = 0.0

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
                    t_c = t_obs - t_obs.mean()
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
    os.makedirs("results", exist_ok=True)

    print("Loading canonical split...")
    split    = np.load(SPLIT_PATH)
    test_ids = set(split["test_ids"].tolist())

    print("Loading structured data...")
    s         = np.load(STRUCTURED_PATH)
    ts_ids    = s["stay_ids"].astype(int)
    ts_values = s["values"]
    ts_masks  = s["masks"]
    ts_labels = s["labels"].astype(int)

    # Select test patients
    mask        = np.array([sid in test_ids for sid in ts_ids])
    test_values = ts_values[mask]
    test_masks  = ts_masks[mask]
    test_labels = ts_labels[mask]
    print(f"Test patients : {test_values.shape[0]:,}  mortality : {test_labels.mean():.4f}")

    print("Building features...")
    X_test = extract_features(test_values, test_masks)

    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Running inference...")
    preds = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(test_labels, preds)
    auprc = average_precision_score(test_labels, preds)
    print(f"XGBoost  AUROC : {auroc:.4f}  AUPRC : {auprc:.4f}")

    np.savez(SAVE_PATH, preds=preds, labels=test_labels)
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
