"""
PhysioNet 2012 Challenge - Data loading and preprocessing.

Each patient file has format: Time,Parameter,Value
Time is HH:MM, Parameter is a clinical variable name, Value is the reading.
We convert to irregularly-sampled time series tensors.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle

# 37 clinical variables used in the challenge
VARIABLES = [
    'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'MAP', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'pH'
]
# Deduplicate while preserving order
seen = set()
VARIABLES = [v for v in VARIABLES if not (v in seen or seen.add(v))]
VAR_INDEX = {v: i for i, v in enumerate(VARIABLES)}
N_VARS = len(VARIABLES)


def parse_patient(filepath):
    """
    Parse one patient .txt file into arrays of (times, values, mask).
    
    Returns:
        times  : np.array [T] — observation times in hours (0-48)
        values : np.array [T, N_VARS] — observed values (0 if missing)
        mask   : np.array [T, N_VARS] — 1 if observed, 0 if missing
        static : dict — static variables (Age, Gender, Height, Weight, ICUType)
    """
    df = pd.read_csv(filepath)
    
    # Parse time to hours
    def time_to_hours(t):
        h, m = map(int, t.split(':'))
        return h + m / 60.0
    
    # Extract static variables from first rows (time=00:00)
    static_rows = df[df['Time'] == '00:00']
    static = {}
    for _, row in static_rows.iterrows():
        static[row['Parameter']] = row['Value']
    
    # Remove static + RecordID rows
    static_params = {'RecordID', 'Age', 'Gender', 'Height', 'Weight', 'ICUType'}
    df = df[~df['Parameter'].isin(static_params)].copy()
    df = df[df['Parameter'].isin(VAR_INDEX)].copy()
    
    if df.empty:
        return None, None, None, static
    
    df['hours'] = df['Time'].apply(time_to_hours)
    
    # Get unique time points
    time_points = sorted(df['hours'].unique())
    T = len(time_points)
    time_idx = {t: i for i, t in enumerate(time_points)}
    
    times = np.array(time_points, dtype=np.float32)
    values = np.zeros((T, N_VARS), dtype=np.float32)
    mask = np.zeros((T, N_VARS), dtype=np.float32)
    
    for _, row in df.iterrows():
        if row['Parameter'] in VAR_INDEX and row['Value'] != -1:
            t_idx = time_idx[row['hours']]
            v_idx = VAR_INDEX[row['Parameter']]
            values[t_idx, v_idx] = row['Value']
            mask[t_idx, v_idx] = 1.0
    
    return times, values, mask, static


def load_set(data_dir, outcomes_file, max_patients=None):
    """Load all patients from a set directory."""
    outcomes = pd.read_csv(outcomes_file)
    outcomes = outcomes.set_index('RecordID')
    
    patients = []
    files = sorted(os.listdir(data_dir))
    if max_patients:
        files = files[:max_patients]
    
    for fname in files:
        if not fname.endswith('.txt'):
            continue
        record_id = int(fname.replace('.txt', ''))
        if record_id not in outcomes.index:
            continue
        
        label = int(outcomes.loc[record_id, 'In-hospital_death'])
        filepath = os.path.join(data_dir, fname)
        times, values, mask, static = parse_patient(filepath)
        
        if times is None or len(times) < 2:
            continue
        
        patients.append({
            'record_id': record_id,
            'times': times,
            'values': values,
            'mask': mask,
            'static': static,
            'label': label,
        })
    
    return patients


def collate_fn(batch):
    """
    Collate variable-length time series into padded tensors.
    
    Returns dict with:
        times       : [B, T_max] padded time points
        values      : [B, T_max, N_VARS]
        mask        : [B, T_max, N_VARS]
        seq_lengths : [B] actual lengths
        labels      : [B]
    """
    batch = sorted(batch, key=lambda x: len(x['times']), reverse=True)
    B = len(batch)
    T_max = max(len(p['times']) for p in batch)
    
    times = torch.zeros(B, T_max)
    values = torch.zeros(B, T_max, N_VARS)
    mask = torch.zeros(B, T_max, N_VARS)
    seq_lengths = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B)
    
    for i, p in enumerate(batch):
        T = len(p['times'])
        times[i, :T] = torch.tensor(p['times'])
        values[i, :T] = torch.tensor(p['values'])
        mask[i, :T] = torch.tensor(p['mask'])
        seq_lengths[i] = T
        labels[i] = p['label']
    
    return {
        'times': times,
        'values': values,
        'mask': mask,
        'seq_lengths': seq_lengths,
        'labels': labels,
    }


class PhysioNetDataset(Dataset):
    def __init__(self, patients):
        self.patients = patients
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        return self.patients[idx]


def preprocess_and_save(raw_dir, processed_dir):
    """Load all sets, normalize, and save to processed/."""
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Loading set-a (train)...")
    train = load_set(
        os.path.join(raw_dir, 'set-a'),
        os.path.join(raw_dir, 'Outcomes-a.txt')
    )
    print(f"  {len(train)} patients loaded")
    
    print("Loading set-b (val)...")
    val = load_set(
        os.path.join(raw_dir, 'set-b'),
        os.path.join(raw_dir, 'Outcomes-b.txt')
    )
    print(f"  {len(val)} patients loaded")
    
    print("Loading set-c (test)...")
    test = load_set(
        os.path.join(raw_dir, 'set-c'),
        os.path.join(raw_dir, 'Outcomes-c.txt')
    )
    print(f"  {len(test)} patients loaded")
    
    # Fit scaler on training set only
    print("Fitting normalizer on train set...")
    all_values = np.concatenate([p['values'].reshape(-1, N_VARS) for p in train], axis=0)
    all_masks = np.concatenate([p['mask'].reshape(-1, N_VARS) for p in train], axis=0)
    
    means = np.zeros(N_VARS)
    stds = np.ones(N_VARS)
    for v in range(N_VARS):
        obs = all_values[all_masks[:, v] == 1, v]
        if len(obs) > 0:
            means[v] = obs.mean()
            stds[v] = obs.std() if obs.std() > 0 else 1.0
    
    # Normalize observed values
    for split in [train, val, test]:
        for p in split:
            for v in range(N_VARS):
                obs_mask = p['mask'][:, v] == 1
                p['values'][obs_mask, v] = (p['values'][obs_mask, v] - means[v]) / stds[v]
    
    # Print class balance
    train_labels = [p['label'] for p in train]
    print(f"Train mortality rate: {sum(train_labels)/len(train_labels):.1%}")
    
    # Save
    with open(os.path.join(processed_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(processed_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val, f)
    with open(os.path.join(processed_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test, f)
    np.save(os.path.join(processed_dir, 'means.npy'), means)
    np.save(os.path.join(processed_dir, 'stds.npy'), stds)
    
    print(f"Saved to {processed_dir}")
    return train, val, test


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base, 'data', 'raw')
    processed_dir = os.path.join(base, 'data', 'processed')
    preprocess_and_save(raw_dir, processed_dir)
