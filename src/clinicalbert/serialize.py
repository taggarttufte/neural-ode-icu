"""
Convert PhysioNet patient time-series data into text for ClinicalBERT.

Each patient becomes a string like:
"At hour 1.5: HR=82, Temp=37.1. At hour 2.3: HR=79, MAP=65. ..."
"""

import numpy as np
from src.dataset import VARIABLES

# Load normalization stats to recover real clinical values
_means = np.load("data/processed/means.npy")
_stds  = np.load("data/processed/stds.npy")


def patient_to_text(times, values, mask, max_tokens=512, denormalize=False):
    """
    Serialize a patient's time series into a clinical text string.

    Args:
        times  : np.array [T]       - observation times in hours
        values : np.array [T, N_VARS] - observed values
        mask   : np.array [T, N_VARS] - 1 if observed, 0 if missing
        max_tokens : int - approximate token budget (BERT max is 512)

    Returns:
        str - serialized patient text
    """
    # Reverse z-score normalization: x_real = x_norm * std + mean
    if denormalize:
        real_values = values * _stds + _means
    else:
        real_values = values

    parts = []

    for t_idx, t in enumerate(times):
        # Get observed variables at this time step
        observed = []
        for v_idx, var_name in enumerate(VARIABLES):
            if mask[t_idx, v_idx] == 1:
                val = real_values[t_idx, v_idx]
                observed.append(f"{var_name}={val:.1f}")

        if observed:
            parts.append(f"At hour {t:.1f}: {', '.join(observed)}.")

    text = " ".join(parts)

    # Rough truncation to stay within BERT's 512 token limit
    # ~4 chars per token on average; leave room for [CLS] and [SEP]
    max_chars = (max_tokens - 10) * 4
    if len(text) > max_chars:
        text = text[:max_chars]

    return text if text else "No observations recorded."
