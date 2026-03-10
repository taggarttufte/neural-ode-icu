#!/usr/bin/env python3
"""
mimic_create_split.py

Creates a single canonical 80/10/10 train/val/test split on the intersection
of structured data (74,829 stays) and notes data (47,142 stays).

This ensures all models are evaluated on IDENTICAL test patients,
making AUROC comparisons statistically valid.

Saves: data/canonical_split.npz
  - train_ids, val_ids, test_ids  : stay_ids for each split
  - train_idx, val_idx, test_idx  : integer indices into the intersection array

Usage:
  python src/mimic_create_split.py
"""

import numpy as np
from sklearn.model_selection import train_test_split

STRUCTURED_PATH = "data/mimic_iv_processed_v2.npz"
NOTES_PATH      = "data/mimic_iv_notes.npz"
SAVE_PATH       = "data/canonical_split.npz"
SEED            = 42


def main():
    print("Loading structured data...")
    s        = np.load(STRUCTURED_PATH)
    ts_ids   = s["stay_ids"].astype(int)
    ts_labels = s["labels"].astype(int)

    print("Loading notes data...")
    n        = np.load(NOTES_PATH, allow_pickle=True)
    note_ids = n["stay_ids"].astype(int)

    # Build intersection
    ts_id_set   = set(ts_ids.tolist())
    note_id_set = set(note_ids.tolist())
    shared_ids  = np.array(sorted(ts_id_set & note_id_set))

    print(f"Structured stays  : {len(ts_ids):,}")
    print(f"Notes stays       : {len(note_ids):,}")
    print(f"Intersection      : {len(shared_ids):,}")

    # Get labels for intersection (use structured data labels)
    ts_id_map     = {sid: i for i, sid in enumerate(ts_ids)}
    shared_labels = np.array([ts_labels[ts_id_map[sid]] for sid in shared_ids])

    print(f"Mortality rate    : {shared_labels.mean():.4f}")

    # Stratified 80 / 10 / 10 split
    idx = np.arange(len(shared_ids))
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.20, random_state=SEED, stratify=shared_labels
    )
    val_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=SEED, stratify=shared_labels[tmp_idx]
    )

    print(f"Train : {len(tr_idx):,}  Val : {len(val_idx):,}  Test : {len(te_idx):,}")
    print(f"Test mortality : {shared_labels[te_idx].mean():.4f}")

    np.savez(
        SAVE_PATH,
        # stay_ids for each split
        train_ids = shared_ids[tr_idx],
        val_ids   = shared_ids[val_idx],
        test_ids  = shared_ids[te_idx],
        # integer positions into the sorted intersection array
        train_idx = tr_idx,
        val_idx   = val_idx,
        test_idx  = te_idx,
        # full sorted intersection (so other scripts can align easily)
        all_ids    = shared_ids,
        all_labels = shared_labels,
    )
    print(f"Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
