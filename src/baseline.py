"""
XGBoost baseline for ICU mortality prediction.

Strategy: extract hand-crafted statistical features from each patient's
time series, then train a gradient boosted classifier. This mirrors the
approach used by top teams in the PhysioNet 2012 Challenge.

Features extracted per variable (36 vars x 10 = 360 features):
  - count      : number of observations
  - mean       : mean of observed values
  - std        : std of observed values
  - min        : minimum observed value
  - max        : maximum observed value
  - first      : first observed value
  - last       : last observed value
  - trend      : slope of obs vs time (linear regression)
  - t_first    : time of first observation
  - t_last     : time of last observation

Global features (2):
  - total_obs  : total observations across all variables
  - miss_rate  : overall missingness rate

Static features (5):
  - Age, Gender, Height, ICUType, Weight

Total: 367 features per patient
"""

import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_VARS = 36


def extract_features(patient):
    """Extract statistical features from a single patient record."""
    times  = np.array(patient['times'])   # [T]
    values = np.array(patient['values'])  # [T, N_VARS]
    mask   = np.array(patient['mask'])    # [T, N_VARS]
    static = patient['static']

    features = []

    # Per-variable features
    for v in range(N_VARS):
        obs_mask = mask[:, v].astype(bool)
        obs_vals = values[obs_mask, v]
        obs_times = times[obs_mask]
        count = obs_mask.sum()

        if count == 0:
            # Variable never observed — all zeros
            features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            mean_v  = obs_vals.mean()
            std_v   = obs_vals.std() if count > 1 else 0.0
            min_v   = obs_vals.min()
            max_v   = obs_vals.max()
            first_v = obs_vals[0]
            last_v  = obs_vals[-1]
            t_first = obs_times[0]
            t_last  = obs_times[-1]

            # Trend: slope of linear fit to (time, value)
            if count > 1:
                t_centered = obs_times - obs_times.mean()
                denom = (t_centered ** 2).sum()
                trend = (t_centered * obs_vals).sum() / denom if denom > 0 else 0.0
            else:
                trend = 0.0

            features.extend([count, mean_v, std_v, min_v, max_v,
                              first_v, last_v, trend, t_first, t_last])

    # Global missingness features
    total_obs = mask.sum()
    miss_rate = 1.0 - mask.mean()
    features.extend([total_obs, miss_rate])

    # Static features (skip RecordID)
    for key in ['Age', 'Gender', 'Height', 'ICUType', 'Weight']:
        val = static.get(key, -1.0)
        features.append(val if val is not None else -1.0)

    return np.array(features, dtype=np.float32)


def build_dataset(data):
    """Extract features and labels from a list of patient records."""
    X = np.stack([extract_features(p) for p in data])
    y = np.array([p['label'] for p in data], dtype=np.float32)
    return X, y


def main():
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    results_dir   = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("Loading data...")
    with open(os.path.join(processed_dir, 'train.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(processed_dir, 'val.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open(os.path.join(processed_dir, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)

    print("Extracting features...")
    X_train, y_train = build_dataset(train)
    X_val,   y_val   = build_dataset(val)
    X_test,  y_test  = build_dataset(test)
    print(f"  Feature shape: {X_train.shape}  ({X_train.shape[1]} features per patient)")

    # Handle -1 placeholders for unrecorded static features
    X_train = np.nan_to_num(X_train)
    X_val   = np.nan_to_num(X_val)
    X_test  = np.nan_to_num(X_test)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Class imbalance: weight positives ~6x
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Positive class weight: {scale_pos_weight:.2f}x")

    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        device='cpu',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    for split, X, y in [('Val', X_val, y_val), ('Test', X_test, y_test)]:
        probs = model.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, probs)
        auprc = average_precision_score(y, probs)
        print(f"\n{split} Results:")
        print(f"  AUROC : {auroc:.4f}")
        print(f"  AUPRC : {auprc:.4f}")

    # Feature importance (top 20)
    print("\nTop 20 most important features:")
    var_names = [f"var{v:02d}_{stat}" for v in range(N_VARS)
                 for stat in ['count','mean','std','min','max','first','last','trend','t_first','t_last']]
    var_names += ['total_obs', 'miss_rate', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {var_names[idx]:<30s} {importances[idx]:.4f}")

    # Save model
    model.save_model(os.path.join(results_dir, 'xgboost_baseline.json'))
    print(f"\nModel saved to results/xgboost_baseline.json")


if __name__ == '__main__':
    main()
