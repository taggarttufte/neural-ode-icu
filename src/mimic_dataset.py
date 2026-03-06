"""
MIMICDataset — loads mimic_iv_processed_v2.npz and returns batches
compatible with the existing LatentODE model.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

N_VARS = 20  # 18 chartevents + urine_output + age


class MIMICDataset(Dataset):
    def __init__(self, values, masks, times, labels):
        # values : (N, 48, 20)
        # masks  : (N, 48, 20)
        # times  : (48,)
        # labels : (N,)
        self.values = torch.tensor(values, dtype=torch.float32)
        self.masks   = torch.tensor(masks,  dtype=torch.float32)
        self.times   = torch.tensor(times,  dtype=torch.float32)
        self.labels  = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'values':  self.values[idx],   # (48, 20)
            'mask':    self.masks[idx],    # (48, 20)
            'times':   self.times,         # (48,)
            'labels':  self.labels[idx],   # scalar
        }


def collate_fn(batch):
    values  = torch.stack([b['values']  for b in batch])
    masks   = torch.stack([b['mask']    for b in batch])
    times   = batch[0]['times']
    labels  = torch.stack([b['labels']  for b in batch])
    seq_lengths = masks.sum(dim=1).max(dim=1).values.long()
    return {
        'values':       values,
        'mask':         masks,
        'times':        times.unsqueeze(0).expand(len(batch), -1),
        'seq_lengths':  seq_lengths,
        'labels':       labels,
    }


def load_splits(npz_path, seed=42, train_frac=0.70, val_frac=0.15):
    """Load .npz and return train/val/test MIMICDataset splits."""
    d = np.load(npz_path)
    values = d['values']
    masks  = d['masks']
    times  = d['times']
    labels = d['labels']

    N = len(labels)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]

    print(f"Split: {len(tr)} train / {len(va)} val / {len(te)} test")
    print(f"Mortality — train: {labels[tr].mean():.3f} | "
          f"val: {labels[va].mean():.3f} | test: {labels[te].mean():.3f}")

    train_ds = MIMICDataset(values[tr], masks[tr], times, labels[tr])
    val_ds   = MIMICDataset(values[va], masks[va], times, labels[va])
    test_ds  = MIMICDataset(values[te], masks[te], times, labels[te])
    return train_ds, val_ds, test_ds
