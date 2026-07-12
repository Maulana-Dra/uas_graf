"""
stats_h1.py
===========
Formal statistical re-validation of H1 from raw CSV data.

H1: ICC is >= 80% faster than Full Recompute at churn rate < 10%.

Produces data/h1_stats_formal.csv with effect sizes (Cohen's d).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUTPUT_PATH = _ROOT / "data" / "h1_stats_formal.csv"


def _cohens_d(diff: np.ndarray) -> float:
    """Compute Cohen's d from a paired-difference array."""
    std = diff.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(diff.mean() / std)


def _effect_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    if np.isnan(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def run_stats_h1(
    raw_path: Path | str,
    summary_path: Path | str,
) -> pd.DataFrame:
    """Re-validate H1 from raw CSV with effect sizes.

    Parameters
    ----------
    raw_path : Path or str
        Path to data/h1_raw.csv.
    summary_path : Path or str
        Path to data/h1_summary.csv.

    Returns
    -------
    pd.DataFrame
        Formal stats table (one row per N × churn_rate combo).
    """
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — run experiments/run_h1.py first"
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found — run experiments/run_h1.py first"
        )

    df_raw = pd.read_csv(raw_path)
    df_sum = pd.read_csv(summary_path)

    rows: list[dict] = []

    for (N, churn_rate), grp in df_raw.groupby(["N", "churn_rate"]):
        fr_times = grp["fr_elapsed_ms"].values.astype(float)
        icc_times = grp["icc_elapsed_ms"].values.astype(float)
        diff = fr_times - icc_times

        # Guard: need at least 3 observations for valid stats
        if len(diff) < 3:
            warnings.warn(
                f"H1 group N={N} churn={churn_rate}: "
                f"only {len(diff)} observations — stats may be unreliable."
            )

        # ---- Shapiro-Wilk normality on differences ----
        if len(diff) >= 3:
            try:
                stat_sw, p_sw = shapiro(diff)
            except Exception:
                stat_sw, p_sw = float("nan"), 0.0
        else:
            stat_sw, p_sw = float("nan"), 0.0

        is_normal = (p_sw >= 0.05) and not np.isnan(p_sw)

        # ---- Primary statistical test ----
        if is_normal and len(diff) >= 3:
            try:
                test_stat, p_val = ttest_rel(fr_times, icc_times)
                test_used = "Paired T-Test"
            except Exception:
                test_stat, p_val = float("nan"), float("nan")
                test_used = "Paired T-Test (failed)"
        else:
            try:
                test_stat, p_val = wilcoxon(fr_times, icc_times)
                test_used = "Wilcoxon"
            except Exception:
                test_stat, p_val = float("nan"), float("nan")
                test_used = "Wilcoxon (failed)"

        # ---- Effect size: Cohen's d ----
        d = _cohens_d(diff)

        # ---- Join h1_criterion from summary ----
        match = df_sum[
            (df_sum["N"] == N) & (df_sum["churn_rate"] == churn_rate)
        ]
        h1_crit = match["h1_criterion"].iloc[0] if len(match) > 0 else "N/A"
        mean_eff = match["mean_efficiency_pct"].iloc[0] if len(match) > 0 else float("nan")
        std_eff = match["std_efficiency_pct"].iloc[0] if len(match) > 0 else float("nan")

        rows.append({
            "N": N,
            "churn_rate": churn_rate,
            "n_batches": len(grp),
            "mean_fr_ms": float(np.mean(fr_times)),
            "mean_icc_ms": float(np.mean(icc_times)),
            "mean_efficiency_pct": float(mean_eff),
            "std_efficiency_pct": float(std_eff),
            "shapiro_p": float(p_sw),
            "test_used": test_used,
            "test_stat": float(test_stat) if not np.isnan(float(test_stat) if test_stat is not None else float("nan")) else float("nan"),
            "p_value": float(p_val) if not np.isnan(float(p_val) if p_val is not None else float("nan")) else float("nan"),
            "significant": bool((float(p_val) < 0.05) if (p_val is not None and not np.isnan(float(p_val))) else False),
            "cohens_d": d,
            "effect_size": _effect_label(d),
            "h1_criterion": h1_crit,
        })

    df_out = pd.DataFrame(rows)

    # ---- Console table ----
    print("\n  H1 FORMAL STATISTICS")
    print("  " + "-" * 95)
    hdr = (
        f"  {'N':>8} {'Churn':>6} {'Batches':>7} "
        f"{'Eff%':>8} {'p-val':>8} {'Test':>16} "
        f"{'Cohen d':>9} {'Effect':>10} {'Criterion':>15}"
    )
    print(hdr)
    print("  " + "-" * 95)
    for _, r in df_out.iterrows():
        sig = "*" if r["significant"] else " "
        pv = f"{r['p_value']:.4f}{sig}" if not np.isnan(r["p_value"]) else "N/A"
        cd = f"{r['cohens_d']:.3f}" if not np.isnan(r["cohens_d"]) else "N/A"
        eff = f"{r['mean_efficiency_pct']:.1f}%" if not np.isnan(r["mean_efficiency_pct"]) else "N/A"
        print(
            f"  {int(r['N']):>8,} {r['churn_rate']:>6.2f} {int(r['n_batches']):>7} "
            f"  {eff:>8} {pv:>8} {r['test_used']:>16} "
            f"  {cd:>9} {r['effect_size']:>10} {r['h1_criterion']:>15}"
        )
    print("  " + "-" * 95)
    print("  (* = p < 0.05)")

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved: {OUTPUT_PATH}")
    return df_out


def print_h1_verdict(df: pd.DataFrame) -> None:
    """Print overall H1 verdict based on formal stats DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of run_stats_h1().
    """
    if df.empty:
        print("\n  H1: Tidak cukup data untuk menentukan verdict.")
        return

    n_meets = (df["h1_criterion"] == "MEETS_TARGET").sum()
    n_total = len(df)

    low_churn = df[df["churn_rate"] < 0.10]
    high_churn = df[df["churn_rate"] >= 0.10]

    low_meets = (low_churn["h1_criterion"] == "MEETS_TARGET").all() if len(low_churn) > 0 else False
    high_meets = (high_churn["h1_criterion"] == "MEETS_TARGET").all() if len(high_churn) > 0 else False
    high_sig = high_churn["significant"].any() if len(high_churn) > 0 else False

    print("\n  === H1 VERDICT ===")
    if low_meets and high_meets:
        print(
            "  H1 DITERIMA PENUH: ICC mencapai efisiensi >=80% pada "
            "semua tingkat churn yang diuji (1%, 5%, 10%)."
        )
    elif low_meets and high_sig:
        print(
            "  H1 DITERIMA SEBAGIAN: ICC mencapai efisiensi >=80% "
            "pada churn <=5%, masih signifikan pada churn 10% "
            "meskipun belum mencapai target 80%."
        )
    elif low_meets:
        print(
            "  H1 DITERIMA SEBAGIAN: ICC mencapai efisiensi >=80% "
            "pada churn <=5%, namun tidak signifikan pada churn 10%."
        )
    else:
        print(
            f"  H1 DITOLAK: ICC hanya memenuhi target pada "
            f"{n_meets}/{n_total} kombinasi. "
            "Lihat analisis n_affected untuk penjelasan."
        )
    print(
        f"  Kombinasi memenuhi target (>=80%): {n_meets}/{n_total}\n"
        f"  Catatan: Tingginya n_affected pada churn besar "
        "menyebabkan ICC mendekati kecepatan FR (2-hop saturation)."
    )
