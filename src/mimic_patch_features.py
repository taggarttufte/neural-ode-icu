#!/usr/bin/env python3
"""
Patch script: adds Age and Urine Output to existing mimic_iv_processed.npz
without reprocessing chartevents.

New output: data/mimic_iv_processed_v2.npz
  - values : (N, 48, 20) — 18 original + urine_output + age (constant)
  - masks  : (N, 48, 20)
  - times, labels, stay_ids: unchanged
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE     = Path("/home/j19w245/data/mimic-iv/physionet.org/files/mimiciv/3.1")
ICU_DIR  = BASE / "icu"
HOSP_DIR = BASE / "hosp"
IN_PATH  = Path("/home/j19w245/neural-ode-icu/data/mimic_iv_processed.npz")
OUT_PATH = Path("/home/j19w245/neural-ode-icu/data/mimic_iv_processed_v2.npz")

# Urine output itemids (all urinary output sources in MIMIC-IV)
URINE_ITEMIDS = {226559, 226560, 226561, 226584, 226563,
                 226564, 226565, 226567, 226557, 226558}

N_BINS = 48


def main():
    # ── Load existing processed data ──────────────────────────────────────
    print("Loading existing processed data...")
    d = np.load(IN_PATH)
    values   = d['values']    # (N, 48, 18)
    masks    = d['masks']
    times    = d['times']
    labels   = d['labels']
    stay_ids = d['stay_ids']
    N = len(stay_ids)
    print(f"  {N} patients, {values.shape[2]} features")

    stay_pos = {sid: i for i, sid in enumerate(stay_ids)}

    # ── Load ICU stays for subject_id <-> stay_id mapping ─────────────────
    print("Loading ICU stays for subject_id mapping...")
    stays = pd.read_csv(ICU_DIR / "icustays.csv.gz",
                        usecols=["stay_id", "subject_id"],
                        parse_dates=["intime"] if False else None)
    stays = stays[stays["stay_id"].isin(set(stay_ids))]

    # ── Age from patients table ───────────────────────────────────────────
    print("Loading patients for age...")
    patients = pd.read_csv(HOSP_DIR / "patients.csv.gz",
                           usecols=["subject_id", "anchor_age"])
    stays = stays.merge(patients, on="subject_id", how="left")

    age_col = np.zeros(N, dtype=np.float32)
    for _, row in stays.iterrows():
        if row["stay_id"] in stay_pos:
            age_col[stay_pos[row["stay_id"]]] = row["anchor_age"]

    # Z-score age
    mu, std = age_col.mean(), age_col.std() + 1e-8
    age_col = (age_col - mu) / std

    # Broadcast age as constant across all 48 time bins
    age_ts   = np.tile(age_col[:, None, None], (1, N_BINS, 1))  # (N,48,1)
    age_mask = np.ones((N, N_BINS, 1), dtype=np.float32)

    # ── Urine output from outputevents ────────────────────────────────────
    print("Loading outputevents for urine output...")
    stays_intime = pd.read_csv(ICU_DIR / "icustays.csv.gz",
                               usecols=["stay_id", "intime"],
                               parse_dates=["intime"])
    stays_intime = stays_intime[stays_intime["stay_id"].isin(set(stay_ids))]
    intime_map = dict(zip(stays_intime["stay_id"], stays_intime["intime"]))

    outputs = pd.read_csv(ICU_DIR / "outputevents.csv.gz",
                          usecols=["stay_id", "itemid", "charttime", "value"],
                          parse_dates=["charttime"],
                          dtype={"stay_id": "int32", "itemid": "int32",
                                 "value": "float32"})
    outputs = outputs[outputs["itemid"].isin(URINE_ITEMIDS)]
    outputs = outputs[outputs["stay_id"].isin(set(stay_ids))]
    outputs = outputs.dropna(subset=["value"]).reset_index(drop=True)
    print(f"  {len(outputs)} urine output rows")

    urine_vals = np.zeros((N, N_BINS, 1), dtype=np.float32)
    urine_mask = np.zeros((N, N_BINS, 1), dtype=np.float32)

    outputs["intime"] = outputs["stay_id"].map(intime_map)
    outputs["hours"]  = (outputs["charttime"] - outputs["intime"]).dt.total_seconds() / 3600
    outputs = outputs[(outputs["hours"] >= 0) & (outputs["hours"] < N_BINS)]
    outputs["bin"]    = outputs["hours"].astype(int)
    outputs["stay_i"] = outputs["stay_id"].map(stay_pos)
    outputs = outputs.dropna(subset=["stay_i"])
    outputs["stay_i"] = outputs["stay_i"].astype(int)

    # Sum urine output within each hour bin
    grouped = outputs.groupby(["stay_i", "bin"])["value"].sum().reset_index()
    for row in grouped.itertuples(index=False):
        urine_vals[row.stay_i, row.bin, 0] = row.value
        urine_mask[row.stay_i, row.bin, 0] = 1.0

    # Z-score urine output
    mu  = urine_vals[urine_mask == 1].mean() if urine_mask.sum() > 0 else 0
    std = urine_vals[urine_mask == 1].std() + 1e-8 if urine_mask.sum() > 0 else 1
    urine_vals = (urine_vals - mu) / std

    # ── Concatenate and save ──────────────────────────────────────────────
    print("Concatenating features...")
    values_new = np.concatenate([values, urine_vals, age_ts], axis=2)  # (N,48,20)
    masks_new  = np.concatenate([masks,  urine_mask, age_mask], axis=2)

    np.savez(OUT_PATH, values=values_new, masks=masks_new, times=times,
             labels=labels, stay_ids=stay_ids)

    print(f"\nDone.")
    print(f"  New shape : {values_new.shape}")
    print(f"  Saved to  : {OUT_PATH}")
    print(f"  Features  : 18 original + urine_output + age")


if __name__ == "__main__":
    main()
