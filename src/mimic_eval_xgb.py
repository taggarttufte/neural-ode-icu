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


def build_features(data):
    """
    data : (N, 48, 20) time-series array
    Returns (N, 20*6) feature matrix:
      for each variable: mean, std, min, max, last-observed, missing-rate
    """
    N, T, V = data.shape
    feats = []
    for v in range(V):
        x = data[:, :, v]                  # (N, T)
        obs_mask = (x != 0).astype(float)  # rough missingness proxy

        mean_val  = x.mean(axis=1)
        std_val   = x.std(axis=1)
        min_val   = x.min(axis=1)
        max_val   = x.max(axis=1)
        miss_rate = 1 - obs_mask.mean(axis=1)

        # last observed (non-zero) value; fall back to 0 if all missing
        last_val = np.zeros(N)
        for i in range(N):
            nz = np.where(obs_mask[i] > 0)[0]
            if len(nz) > 0:
                last_val[i] = x[i, nz[-1]]

        feats.append(np.stack([mean_val, std_val, min_val, max_val,
                                last_val, miss_rate], axis=1))

    return np.concatenate(feats, axis=1)   # (N, V*6)


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading canonical split...")
    split    = np.load(SPLIT_PATH)
    test_ids = set(split["test_ids"].tolist())

    print("Loading structured data...")
    s        = np.load(STRUCTURED_PATH)
    ts_ids   = s["stay_ids"].astype(int)
    ts_data  = s["data"]
    ts_labels = s["labels"].astype(int)

    # Select test patients
    mask      = np.array([sid in test_ids for sid in ts_ids])
    test_data = ts_data[mask]
    test_labels = ts_labels[mask]
    print(f"Test patients : {test_data.shape[0]:,}  mortality : {test_labels.mean():.4f}")

    print("Building features...")
    X_test = build_features(test_data)

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
