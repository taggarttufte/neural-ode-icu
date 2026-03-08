#!/usr/bin/env python3
"""
Preprocessing script: MIMIC-IV clinical notes for ClinicalBERT.

Matches notes to the same 74,829 ICU stays in mimic_iv_processed_v2.npz.

Strategy (avoids data leakage):
  - Use ALL note types written within the first 48h of ICU admission
  - Discharge summaries are EXCLUDED (written after outcome is known)
  - If a stay has no notes in the first 48h, we mark it as missing
    and it is dropped from the notes dataset.

Outputs:
  data/mimic_iv_notes.npz
    texts     : (M,) object array of strings
    labels    : (M,) float32 array (in-hospital mortality)
    stay_ids  : (M,) int32 array
    note_count: (M,) int32 array (number of notes concatenated per patient)

Where M <= 74,829 (some stays have no notes in the first 48h).

Usage:
    python src/mimic_prep_notes.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
NOTE_DIR   = Path("/home/j19w245/data/mimic-iv/physionet.org/files/mimiciv-note/2.2/note")
ICU_DIR    = Path("/home/j19w245/data/mimic-iv/physionet.org/files/mimiciv/3.1/icu")
PROCESSED  = Path("/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz")
OUT_PATH   = Path("/home/j19w245/neural-ode-icu/data/mimic_iv_notes.npz")

# Exclude discharge summaries — they're written after the ICU stay ends
# and implicitly encode the outcome (alive vs expired).
EXCLUDE_NOTE_TYPES = {'Discharge summary', 'Discharge Summary'}

# How many hours into the ICU stay to include notes from
NOTE_WINDOW_HOURS = 48

# Max chars per patient (BERT truncates to 512 tokens anyway, but keep RAM sane)
MAX_TEXT_CHARS = 8000


def main():
    # ── 1. Load our existing stay_ids and labels ──────────────────────────
    print("Loading processed data...")
    d = np.load(PROCESSED)
    stay_ids_all = d['stay_ids'].astype(int)
    labels_all   = d['labels'].astype(np.float32)
    print(f"  {len(stay_ids_all)} stays in processed dataset")

    # ── 2. Load ICU stays to get intime + hadm_id ─────────────────────────
    print("Loading ICU stays...")
    stays = pd.read_csv(ICU_DIR / "icustays.csv.gz",
                        usecols=['stay_id', 'hadm_id', 'subject_id', 'intime'],
                        parse_dates=['intime'])
    stays = stays[stays['stay_id'].isin(stay_ids_all)].copy()
    stays['intime'] = pd.to_datetime(stays['intime'], utc=False)
    print(f"  {len(stays)} matching stays found in icustays")

    # Map stay_id -> (hadm_id, intime, label)
    label_map   = dict(zip(stay_ids_all, labels_all))
    stays['label'] = stays['stay_id'].map(label_map)
    stays = stays.dropna(subset=['label'])

    # A hospital admission (hadm_id) can contain multiple ICU stays.
    # Keep the first ICU stay per admission so hadm_id is unique.
    stays = stays.sort_values('intime').drop_duplicates('hadm_id', keep='first')
    print(f"  {len(stays)} unique admissions after dedup")

    hadm_to_stay  = stays.set_index('hadm_id')[['stay_id', 'intime', 'label']].to_dict('index')

    # ── 3. Load notes ──────────────────────────────────────────────────────
    # Try discharge.csv.gz first; radiology.csv.gz has separate file
    note_files = list(NOTE_DIR.glob("*.csv.gz"))
    print(f"Note files found: {[f.name for f in note_files]}")

    all_notes = []
    for nf in note_files:
        print(f"  Loading {nf.name}...")
        try:
            df = pd.read_csv(nf, usecols=lambda c: c in
                             ['note_id', 'subject_id', 'hadm_id', 'note_type',
                              'charttime', 'storetime', 'text'],
                             low_memory=False)
            all_notes.append(df)
            print(f"    {len(df)} notes loaded")
        except Exception as e:
            print(f"    Warning: could not load {nf.name}: {e}")

    if not all_notes:
        raise RuntimeError(f"No note files found in {NOTE_DIR}")

    notes = pd.concat(all_notes, ignore_index=True)
    print(f"Total notes loaded: {len(notes)}")

    # ── 4. Filter to our hadm_ids ─────────────────────────────────────────
    notes = notes[notes['hadm_id'].isin(hadm_to_stay)].copy()
    print(f"Notes matching our stays: {len(notes)}")

    # ── 5. Exclude discharge summaries ───────────────────────────────────
    if 'note_type' in notes.columns:
        before = len(notes)
        notes = notes[~notes['note_type'].isin(EXCLUDE_NOTE_TYPES)]
        print(f"After removing discharge summaries: {len(notes)} (removed {before - len(notes)})")

    # ── 6. Filter to first 48h of ICU stay ───────────────────────────────
    # Use charttime if available, otherwise storetime
    notes['time_col'] = pd.to_datetime(
        notes['charttime'].fillna(notes['storetime']), utc=False, errors='coerce'
    )
    notes = notes.dropna(subset=['time_col'])

    # Add intime for each note
    notes['intime'] = notes['hadm_id'].map(
        {k: v['intime'] for k, v in hadm_to_stay.items()}
    )
    notes['hours_from_admit'] = (
        notes['time_col'] - notes['intime']
    ).dt.total_seconds() / 3600

    notes = notes[
        (notes['hours_from_admit'] >= 0) &
        (notes['hours_from_admit'] <= NOTE_WINDOW_HOURS)
    ].copy()
    print(f"Notes within first {NOTE_WINDOW_HOURS}h: {len(notes)}")

    # ── 7. Aggregate notes per stay ──────────────────────────────────────
    # Sort by time, concatenate texts
    notes = notes.sort_values(['hadm_id', 'hours_from_admit'])
    notes['text'] = notes['text'].fillna('').str.strip()

    def aggregate(group):
        texts = group['text'].tolist()
        combined = ' '.join(texts)
        return pd.Series({
            'text':       combined[:MAX_TEXT_CHARS],
            'note_count': len(texts),
        })

    print("Aggregating notes per admission...")
    agg = notes.groupby('hadm_id').apply(aggregate).reset_index()
    print(f"Admissions with early notes: {len(agg)}")

    # ── 8. Build final arrays ─────────────────────────────────────────────
    # Merge back to get stay_id and label
    agg['stay_id'] = agg['hadm_id'].map(
        {k: v['stay_id'] for k, v in hadm_to_stay.items()}
    )
    agg['label'] = agg['hadm_id'].map(
        {k: v['label'] for k, v in hadm_to_stay.items()}
    )
    agg = agg.dropna(subset=['stay_id', 'label'])

    texts      = agg['text'].values
    labels_out = agg['label'].values.astype(np.float32)
    sids_out   = agg['stay_id'].values.astype(np.int32)
    ncounts    = agg['note_count'].values.astype(np.int32)

    mortality_rate = labels_out.mean()
    print(f"\nFinal dataset: {len(texts)} stays with early notes")
    print(f"Mortality rate: {mortality_rate:.1%}")
    print(f"Stays dropped (no early notes): {len(stay_ids_all) - len(texts)}")
    print(f"Avg notes per stay: {ncounts.mean():.1f}")
    print(f"Avg text length: {np.mean([len(t) for t in texts]):.0f} chars")

    # ── 9. Save ───────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        texts=texts,
        labels=labels_out,
        stay_ids=sids_out,
        note_count=ncounts,
    )
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
