#!/usr/bin/env python3
"""
mimic_bootstrap_ci.py

Computes bootstrap 95% confidence intervals and DeLong pairwise tests
for all models evaluated on the canonical test split.

Requires (all from canonical test split):
  - results/preds_xgb.npz
  - results/preds_ode.npz
  - results/preds_bert_notes.npz

Saves:
  - results/bootstrap_ci_results.json   (CIs + DeLong p-values)
  - results/plots/compare_roc_ci.png    (ROC curves with CI bands)
  - results/plots/compare_summary_ci.png (bar chart with error bars)

Usage:
  python src/mimic_bootstrap_ci.py
"""

import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

N_BOOTSTRAP = 2000
SEED        = 42

MODELS = {
    "XGBoost":          "results/preds_xgb.npz",
    "Neural ODE":       "results/preds_ode.npz",
    "Multimodal":       "results/preds_multimodal.npz",
    "BERT (notes)":     "results/preds_bert_notes.npz",
}

COLORS = {
    "XGBoost":      "#4C9BE8",
    "Neural ODE":   "#4ECDC4",
    "Multimodal":   "#A855F7",
    "BERT (notes)": "#FF8C42",
}


# ── DeLong AUROC test ─────────────────────────────────────────────────────────
def structural_components(preds, labels):
    """
    Compute structural components V10 and V01 for the DeLong test.
    Returns (theta, V10, V01) where theta = AUROC.
    """
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    n1, n0 = len(pos), len(neg)

    # Kernel: psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y
    # V10[i] = mean_j psi(pos[i], neg[j])  shape (n1,)
    # V01[j] = mean_i psi(pos[i], neg[j])  shape (n0,)
    psi = (pos[:, None] > neg[None, :]).astype(float) + \
          0.5 * (pos[:, None] == neg[None, :]).astype(float)   # (n1, n0)

    V10 = psi.mean(axis=1)   # (n1,)
    V01 = psi.mean(axis=0)   # (n0,)
    theta = psi.mean()

    return theta, V10, V01


def delong_test(preds1, preds2, labels):
    """
    DeLong et al. (1988) test for equality of two AUROCs on the same test set.
    Returns (auroc1, auroc2, z_stat, p_value).
    """
    from scipy import stats

    theta1, V10_1, V01_1 = structural_components(preds1, labels)
    theta2, V10_2, V01_2 = structural_components(preds2, labels)

    n1 = (labels == 1).sum()
    n0 = (labels == 0).sum()

    # Covariance matrix of (theta1, theta2)
    S10 = np.cov(np.stack([V10_1, V10_2]))   # (2, 2)
    S01 = np.cov(np.stack([V01_1, V01_2]))   # (2, 2)
    S   = S10 / n1 + S01 / n0               # (2, 2)

    # Variance of (theta1 - theta2)
    L   = np.array([1, -1])
    var = L @ S @ L
    z   = (theta1 - theta2) / np.sqrt(var)
    p   = 2 * (1 - stats.norm.cdf(abs(z)))

    return float(theta1), float(theta2), float(z), float(p)


# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_auroc_ci(preds, labels, n=N_BOOTSTRAP, seed=SEED):
    rng   = np.random.default_rng(seed)
    aucs  = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        if labels[idx].sum() == 0 or labels[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(labels[idx], preds[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), aucs


def bootstrap_auprc_ci(preds, labels, n=N_BOOTSTRAP, seed=SEED):
    rng   = np.random.default_rng(seed)
    aucs  = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        if labels[idx].sum() == 0:
            continue
        aucs.append(average_precision_score(labels[idx], preds[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_roc_with_ci(model_data, save_path):
    """model_data: dict of {name: (preds, labels, auroc, lo, hi, color)}"""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, label="Random")

    for name, (preds, labels, auroc, lo, hi, color) in model_data.items():
        fpr, tpr, _ = roc_curve(labels, preds)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  AUROC = {auroc:.4f} [{lo:.4f}–{hi:.4f}]")

    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("MIMIC-IV — ROC Curve Comparison (95% CI)", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    legend = ax.legend(loc="lower right", framealpha=0.2)
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"Saved → {save_path}")


def plot_summary_ci(results, save_path):
    """Bar chart with 95% CI error bars."""
    names  = list(results.keys())
    aurocs = [results[n]["auroc"] for n in names]
    lo_err = [results[n]["auroc"] - results[n]["auroc_ci_lo"] for n in names]
    hi_err = [results[n]["auroc_ci_hi"] - results[n]["auroc"] for n in names]
    colors = [COLORS.get(n, "#aaaaaa") for n in names]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    x = np.arange(len(names))
    bars = ax.bar(x, aurocs, color=colors, width=0.5, alpha=0.85,
                  yerr=[lo_err, hi_err], capsize=6,
                  error_kw={"ecolor": "white", "elinewidth": 1.5})

    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", color="white", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white", fontsize=11)
    ax.set_ylabel("AUROC", color="white", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("MIMIC-IV — AUROC with 95% Bootstrap CI", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"Saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results/plots", exist_ok=True)

    # Load predictions
    loaded = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        d = np.load(path)
        loaded[name] = (d["preds"], d["labels"])
        print(f"Loaded {name}: {len(d['labels']):,} patients")

    if len(loaded) == 0:
        raise RuntimeError("No prediction files found. Run eval scripts first.")

    # Bootstrap CIs
    results    = {}
    model_data = {}
    for name, (preds, labels) in loaded.items():
        auroc     = roc_auc_score(labels, preds)
        auprc     = average_precision_score(labels, preds)
        lo, hi, _ = bootstrap_auroc_ci(preds, labels)
        prc_lo, prc_hi = bootstrap_auprc_ci(preds, labels)

        print(f"{name:20s}  AUROC {auroc:.4f} [{lo:.4f}–{hi:.4f}]  "
              f"AUPRC {auprc:.4f} [{prc_lo:.4f}–{prc_hi:.4f}]")

        results[name] = {
            "auroc": auroc, "auroc_ci_lo": lo, "auroc_ci_hi": hi,
            "auprc": auprc, "auprc_ci_lo": prc_lo, "auprc_ci_hi": prc_hi,
            "n_test": int(len(labels)), "mortality": float(labels.mean()),
        }
        model_data[name] = (preds, labels, auroc, lo, hi, COLORS.get(name, "#aaaaaa"))

    # DeLong pairwise tests
    print("\n── DeLong Pairwise Tests ──────────────────────────────────────────")
    names  = list(loaded.keys())
    delong = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            p1, l1 = loaded[n1]
            p2, l2 = loaded[n2]
            # Must be same test set — use labels from first model
            if not np.array_equal(l1, l2):
                print(f"  WARNING: {n1} vs {n2} have different test labels — skipping")
                continue
            a1, a2, z, p = delong_test(p1, p2, l1)
            key = f"{n1} vs {n2}"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {key:40s}  z={z:+.3f}  p={p:.4f}  {sig}")
            delong[key] = {"auroc_1": a1, "auroc_2": a2, "z": z, "p_value": p}

    results["delong_tests"] = delong

    # Save JSON
    json_path = "results/bootstrap_ci_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # Plots (exclude delong_tests key from summary plot)
    plot_results = {k: v for k, v in results.items() if k != "delong_tests"}
    plot_roc_with_ci(model_data, "results/plots/compare_roc_ci.png")
    plot_summary_ci(plot_results, "results/plots/compare_summary_ci.png")


if __name__ == "__main__":
    main()
