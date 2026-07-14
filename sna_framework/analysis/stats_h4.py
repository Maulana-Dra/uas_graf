"""
stats_h4.py
===========
Formal statistical re-validation of H4 from raw CSV data.

H4: Closeness-seeded information propagation reaches target coverage
    faster than Degree-seeded under random node failure conditions.

Produces data/h4_stats_formal.csv.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUTPUT_PATH = _ROOT / "data" / "h4_stats_formal.csv"


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    """Compute pooled standard deviation for two independent samples."""
    n1, n2 = len(a), len(b)
    if n1 + n2 - 2 <= 0:
        return float("nan")
    pooled = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
        / (n1 + n2 - 2)
    )
    return float(pooled)


def run_stats_h4(
    raw_path: Path | str,
    summary_path: Path | str,
) -> pd.DataFrame:
    """Re-validate H4 from raw CSV with effect sizes.

    Parameters
    ----------
    raw_path : Path or str
        Path to data/h4_raw.csv.
    summary_path : Path or str
        Path to data/h4_summary.csv.

    Returns
    -------
    pd.DataFrame
        Formal stats table (one row per N × failure_rate combo).
    """
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — run experiments/run_h4.py first"
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found — run experiments/run_h4.py first"
        )

    df_raw = pd.read_csv(raw_path)
    df_sum = pd.read_csv(summary_path)

    rows: list[dict] = []

    for (N, failure_rate), grp in df_raw.groupby(["N", "failure_rate"]):
        arr_c = grp["steps_closeness"].values.astype(float)
        arr_d = grp["steps_degree"].values.astype(float)
        n_valid = len(arr_c)

        if n_valid == 0:
            continue

        # ---- Descriptive stats ----
        mean_c, std_c, median_c = float(np.mean(arr_c)), float(np.std(arr_c, ddof=1) if n_valid > 1 else 0.0), float(np.median(arr_c))
        mean_d, std_d, median_d = float(np.mean(arr_d)), float(np.std(arr_d, ddof=1) if n_valid > 1 else 0.0), float(np.median(arr_d))

        # ---- Normality ----
        if n_valid >= 3:
            try:
                _, p_c = shapiro(arr_c)
                _, p_d = shapiro(arr_d)
            except Exception:
                p_c, p_d = 0.0, 0.0
        else:
            p_c, p_d = 0.0, 0.0

        both_normal = (p_c >= 0.05) and (p_d >= 0.05)

        # ---- Statistical test ----
        if both_normal and n_valid >= 3:
            try:
                test_stat, p_val = ttest_ind(arr_c, arr_d)
                test_used = "Independent T-Test"
                use_mwu = False
            except Exception:
                test_stat, p_val = float("nan"), float("nan")
                test_used = "Independent T-Test (failed)"
                use_mwu = False
        else:
            use_mwu = True
            try:
                # alternative='less': tests closeness steps stochastically
                # LESS than degree steps (closeness is faster)
                test_stat, p_val = mannwhitneyu(
                    arr_c, arr_d, alternative="less"
                )
                test_used = "Mann-Whitney U (alt=less)"
            except ValueError:
                warnings.warn(
                    f"H4 mannwhitneyu failed at N={N} "
                    f"failure={failure_rate} — p=1.0"
                )
                test_stat, p_val = float("nan"), 1.0
                test_used = "Mann-Whitney U (failed)"

        # ---- Effect size ----
        if use_mwu and not np.isnan(test_stat):
            # Rank-biserial r for Mann-Whitney U
            # Positive r_rb means closeness tends to have fewer steps
            n1, n2 = len(arr_c), len(arr_d)
            U = float(test_stat)
            r_rb = 1.0 - (2.0 * U) / (n1 * n2)
            effect_size_val = float(r_rb)
            effect_size_label = "rank-biserial r"
        else:
            # Cohen's d for T-Test
            ps = _pooled_std(arr_c, arr_d)
            if ps > 0:
                cohens_d = (mean_c - mean_d) / ps
            else:
                cohens_d = float("nan")
            effect_size_val = cohens_d
            effect_size_label = "Cohen's d"

        # ---- Censoring ----
        pct_c_cens = float(grp["closeness_censored"].mean() * 100)
        pct_d_cens = float(grp["degree_censored"].mean() * 100)

        # ---- Join summary columns ----
        match = df_sum[
            (df_sum["N"] == N) & (df_sum["failure_rate"] == failure_rate)
        ]
        pct_same = float(match["pct_same_node"].iloc[0]) if len(match) > 0 else float("nan")
        pct_c_faster = float(match["pct_closeness_faster"].iloc[0]) if len(match) > 0 else float("nan")
        h4_verdict = match["h4_verdict"].iloc[0] if len(match) > 0 else "N/A"

        significant = bool(float(p_val) < 0.05) if not np.isnan(float(p_val)) else False

        rows.append({
            "N": N,
            "failure_rate": failure_rate,
            "n_valid_runs": n_valid,
            "mean_steps_closeness": mean_c,
            "std_steps_closeness": std_c,
            "median_steps_closeness": median_c,
            "mean_steps_degree": mean_d,
            "std_steps_degree": std_d,
            "median_steps_degree": median_d,
            "pct_same_node": pct_same,
            "pct_closeness_faster": pct_c_faster,
            "pct_closeness_censored": pct_c_cens,
            "pct_degree_censored": pct_d_cens,
            "test_used": test_used,
            "test_stat": float(test_stat) if not np.isnan(float(test_stat)) else float("nan"),
            "p_value": float(p_val),
            "significant": significant,
            "effect_size_val": effect_size_val,
            "effect_size_label": effect_size_label,
            "h4_verdict": h4_verdict,
        })

    df_out = pd.DataFrame(rows)

    # ---- Console table ----
    print("\n  H4 FORMAL STATISTICS")
    print("  " + "-" * 100)
    print(
        f"  {'N':>7} {'Fail%':>6} {'n':>4} "
        f"{'Mean C':>9} {'Mean D':>9} "
        f"{'C<D%':>6} {'Same%':>6} "
        f"{'p-val':>8} {'Effect':>10} {'Verdict':>26}"
    )
    print("  " + "-" * 100)
    for _, r in df_out.iterrows():
        sig = "*" if r["significant"] else " "
        pv = f"{r['p_value']:.4f}{sig}" if not np.isnan(r["p_value"]) else "N/A "
        eff = f"{r['effect_size_val']:.3f}" if not np.isnan(r["effect_size_val"]) else "N/A"
        print(
            f"  {int(r['N']):>7,} {r['failure_rate']*100:>5.0f}% {int(r['n_valid_runs']):>4} "
            f"  {r['mean_steps_closeness']:>9.1f} {r['mean_steps_degree']:>9.1f} "
            f"  {r['pct_closeness_faster']:>5.1f}% {r['pct_same_node']:>5.1f}% "
            f"  {pv:>8} {eff:>10} {r['h4_verdict']:>26}"
        )
    print("  " + "-" * 100)
    print("  (* = p < 0.05, Mann-Whitney U alternative='less')")

    # Same-node overlap warning
    high_same = df_out[df_out["pct_same_node"] > 30]
    if len(high_same) > 0:
        print(
            f"\n  [CONFOUND WARNING] {len(high_same)} combo(s) have "
            "pct_same_node > 30%: closeness_seed == degree_seed. "
            "H4 evaluation is confounded in these cases — "
            "report as methodological limitation in BAB V."
        )

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved: {OUTPUT_PATH}")
    return df_out


def print_h4_verdict(df: pd.DataFrame) -> None:
    """Print overall H4 verdict.

    Parameters
    ----------
    df : pd.DataFrame
        Output of run_stats_h4().
    """
    if df.empty:
        print("\n  H4: Tidak cukup data untuk menentukan verdict.")
        return

    verdicts = df["h4_verdict"].tolist()
    n_accepted = verdicts.count("ACCEPTED")
    n_ns = verdicts.count("REJECTED_NOT_SIGNIFICANT")
    n_deg = verdicts.count("REJECTED_DEGREE_FASTER")
    n_total = len(verdicts)

    mean_same = df["pct_same_node"].mean()

    print("\n  === H4 VERDICT ===")
    if n_accepted == n_total:
        print(
            "  H4 DITERIMA: Closeness seed secara signifikan lebih cepat "
            "dari Degree seed pada semua kombinasi (N × failure_rate)."
        )
    elif n_accepted > n_total // 2:
        print(
            f"  H4 DITERIMA SEBAGIAN: Diterima pada {n_accepted}/{n_total} "
            "kombinasi. Keunggulan closeness bersifat kontekstual."
        )
    elif n_accepted > 0:
        print(
            f"  H4 LEMAH: Hanya {n_accepted}/{n_total} kombinasi yang diterima."
        )
    else:
        print(
            "  H4 DITOLAK: Closeness seed tidak secara signifikan lebih "
            "cepat dari Degree seed pada semua skenario yang diuji."
        )

    print(
        f"  Diterima: {n_accepted}  |  "
        f"Tidak Signifikan: {n_ns}  |  "
        f"Degree Lebih Cepat: {n_deg}"
    )
    if not np.isnan(mean_same) and mean_same > 30:
        print(
            f"  [CONFOUND] Rata-rata pct_same_node = {mean_same:.1f}% "
            "(>30%). Node hub BA mendominasi kedua metrik — "
            "catat sebagai limitasi metodologi."
        )


if __name__ == "__main__":
    import os
    raw_csv = os.path.join("data", "h4_raw.csv")
    summary_csv = os.path.join("data", "h4_summary.csv")
    df = run_stats_h4(raw_csv, summary_csv)
    print_h4_verdict(df)
