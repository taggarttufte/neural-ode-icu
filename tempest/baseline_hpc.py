"""
baseline_hpc.py — Tempest HPC version of baseline.py

Changes from local version:
  - --data-dir / --output-dir arguments
  - num_workers=4 for Linux
  - XGBoost GPU support (device='cuda' if available)

Usage on Tempest:
  python baseline_hpc.py --data-dir /scratch/NETID/mimic-iv/processed \
                         --output-dir /scratch/NETID/results
"""

import os
import argparse
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')
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


def build_dataset(data):
    X = np.stack([extract_features(p) for p in data])
    y = np.array([p['label'] for p in data], dtype=np.float32)
    return X, y


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    device_str = 'cuda' if use_gpu else 'cpu'
    print(f"XGBoost device: {device_str}")

    print("Loading data...")
    with open(os.path.join(args.data_dir, 'train.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(args.data_dir, 'val.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open(os.path.join(args.data_dir, 'test.pkl'), 'rb') as f:
        test = pickle.load(f)

    print("Extracting features...")
    X_train, y_train = build_dataset(train)
    X_val,   y_val   = build_dataset(val)
    X_test,  y_test  = build_dataset(test)
    print(f"  Feature shape: {X_train.shape}")

    X_train = np.nan_to_num(X_train)
    X_val   = np.nan_to_num(X_val)
    X_test  = np.nan_to_num(X_test)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

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
        device=device_str,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    for split, X, y in [('Val', X_val, y_val), ('Test', X_test, y_test)]:
        probs = model.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, probs)
        auprc = average_precision_score(y, probs)
        print(f"\n{split} Results:")
        print(f"  AUROC : {auroc:.4f}")
        print(f"  AUPRC : {auprc:.4f}")

    model.save_model(os.path.join(args.output_dir, 'xgboost_baseline.json'))
    print(f"\nModel saved to {args.output_dir}/xgboost_baseline.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    main(args)
