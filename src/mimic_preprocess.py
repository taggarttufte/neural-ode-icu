#!/usr/bin/env python3
"""
MIMIC-IV ICU preprocessing for Neural ODE / XGBoost / ClinicalBERT baselines.

Outputs: data/mimic_iv_processed.npz
  - values : (N, 48, 18) float32 — hourly binned, forward-filled, z-scored
  - masks  : (N, 48, 18) float32 — 1 where observed
  - times  : (48,)       float32 — hours 0..47
  - labels : (N,)        float32 — in-hospital mortality
  - stay_ids: (N,)       int32
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = Path("/home/j19w245/data/mimic-iv/physionet.org/files/mimiciv/3.1")
ICU_DIR  = BASE / "icu"
HOSP_DIR = BASE / "hosp"
OUT_PATH = Path("/home/j19w245/neural-ode-icu/data/mimic_iv_processed.npz")

# ── Feature / ItemID mapping ───────────────────────────────────────────────
ITEMIDS = {
    'heart_rate':   220045,
    'sbp':          220179,
    'dbp':          220180,
    'mbp':          220052,
    'resp_rate':    220210,
    'temp_c':       223762,
    'spo2':         220277,
    'glucose':      220621,
    'creatinine':   220615,
    'sodium':       220645,
    'potassium':    227442,
    'hematocrit':   220545,
    'wbc':          220546,
    'bicarbonate':  227443,
    'bun':          225624,
    'gcs_motor':    223901,
    'gcs_verbal':   223900,
    'gcs_eye':      220739,
}

FEATURE_NAMES  = list(ITEMIDS.keys())
ITEMID_TO_IDX  = {v: i for i, (k, v) in enumerate(ITEMIDS.items())}
ALL_ITEMIDS    = set(ITEMIDS.values())
N_FEATURES     = len(ITEMIDS)
N_BINS         = 48   # hourly bins, first 48h
MIN_LOS_HOURS  = 24


def main():
    # ── 1. ICU stays ──────────────────────────────────────────────────────
    print("Loading ICU stays...")
    stays = pd.read_csv(ICU_DIR / "icustays.csv.gz", parse_dates=["intime", "outtime"])
    stays["los_hours"] = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600
    stays = stays[stays["los_hours"] >= MIN_LOS_HOURS].reset_index(drop=True)
    print(f"  {len(stays)} stays after LOS >= {MIN_LOS_HOURS}h filter")

    # ── 2. Mortality labels from admissions ───────────────────────────────
    print("Loading admissions...")
    adm = pd.read_csv(HOSP_DIR / "admissions.csv.gz", usecols=["hadm_id", "hospital_expire_flag"])
    stays = stays.merge(adm, on="hadm_id", how="left")
    stays = stays.dropna(subset=["hospital_expire_flag"]).reset_index(drop=True)
    print(f"  {len(stays)} stays after label join")
    print(f"  Mortality rate: {stays['hospital_expire_flag'].mean():.3f}")

    stay_id_arr  = stays["stay_id"].values.astype(np.int32)
    intime_map   = dict(zip(stays["stay_id"], stays["intime"]))
    stay_pos_map = {sid: i for i, sid in enumerate(stay_id_arr)}
    valid_stays  = set(stay_id_arr)

    # ── 3. Chartevents ────────────────────────────────────────────────────
    print("Loading chartevents (large file, please wait)...")
    charts = pd.read_csv(
        ICU_DIR / "chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        dtype={"stay_id": "int32", "itemid": "int32", "valuenum": "float32"},
    )
    charts = charts[charts["itemid"].isin(ALL_ITEMIDS)]
    charts = charts[charts["stay_id"].isin(valid_stays)]
    charts = charts.dropna(subset=["valuenum"]).reset_index(drop=True)
    print(f"  {len(charts)} chartevents rows after filter")

    # ── 4. Compute hour bin (vectorised) ──────────────────────────────────
    print("Computing time bins...")
    charts["intime"] = charts["stay_id"].map(intime_map)
    charts["hours"]  = (charts["charttime"] - charts["intime"]).dt.total_seconds() / 3600
    charts = charts[(charts["hours"] >= 0) & (charts["hours"] < N_BINS)].copy()
    charts["bin"]    = charts["hours"].astype(int)
    charts["feat_i"] = charts["itemid"].map(ITEMID_TO_IDX)
    charts["stay_i"] = charts["stay_id"].map(stay_pos_map)

    # ── 5. Fill arrays ────────────────────────────────────────────────────
    print("Building value/mask arrays...")
    N = len(stays)
    values = np.full((N, N_BINS, N_FEATURES), np.nan, dtype=np.float32)
    masks  = np.zeros((N, N_BINS, N_FEATURES), dtype=np.float32)

    # Use last observed value when multiple measurements fall in same bin
    charts_sorted = charts.sort_values("charttime")
    for row in charts_sorted.itertuples(index=False):
        values[row.stay_i, row.bin, row.feat_i] = row.valuenum
        masks[row.stay_i,  row.bin, row.feat_i] = 1.0

    # ── 6. Forward fill within each stay ─────────────────────────────────
    print("Forward filling missing values...")
    for fi in range(N_FEATURES):
        for t in range(1, N_BINS):
            carry = (masks[:, t, fi] == 0) & (~np.isnan(values[:, t - 1, fi]))
            values[carry, t, fi] = values[carry, t - 1, fi]

    # ── 7. Fill remaining NaN with feature median ─────────────────────────
    for fi in range(N_FEATURES):
        col    = values[:, :, fi]
        median = np.nanmedian(col)
        if np.isnan(median):
            median = 0.0
        values[:, :, fi] = np.where(np.isnan(col), median, col)

    # ── 8. Z-score normalise per feature ──────────────────────────────────
    for fi in range(N_FEATURES):
        col = values[:, :, fi]
        mu  = col.mean()
        std = col.std() + 1e-8
        values[:, :, fi] = (col - mu) / std

    # ── 9. Save ───────────────────────────────────────────────────────────
    labels = stays["hospital_expire_flag"].values.astype(np.float32)
    times  = np.arange(N_BINS, dtype=np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, values=values, masks=masks, times=times,
             labels=labels, stay_ids=stay_id_arr)

    print(f"\nDone.")
    print(f"  Shape : {values.shape}")
    print(f"  Mortality rate : {labels.mean():.3f}")
    print(f"  Saved to : {OUT_PATH}")


if __name__ == "__main__":
    main()
