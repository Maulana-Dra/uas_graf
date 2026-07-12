"""
stats_h2.py
===========
Formal statistical summary of H2 accuracy metrics.

H2: LBA with 5% landmarks achieves Pearson r > 0.95 vs Full Recompute
    on graphs with N > 50,000.  (Correlation check, no p-value test.)

Produces data/h2_stats_formal.csv.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUTPUT_PATH = _ROOT / "data" / "h2_stats_formal.csv"


def run_stats_h2(
    raw_path: Path | str,
    summary_path: Path | str,
) -> pd.DataFrame:
    """Produce formal H2 statistics table from raw CSV.

    Parameters
    ----------
    raw_path : Path or str
        Path to data/h2_raw.csv.
    summary_path : Path or str
        Path to data/h2_summary.csv.

    Returns
    -------
    pd.DataFrame
        Formal stats table (one row per N).
    """
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — run experiments/run_h2.py first"
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found — run experiments/run_h2.py first"
        )

    df_raw = pd.read_csv(raw_path)
    df_sum = pd.read_csv(summary_path)

    # Filter only valid runs
    valid_mask = df_raw["h2_criterion"].isin(["MEETS_TARGET", "BELOW_TARGET"])
    df_valid = df_raw[valid_mask].copy()

    rows: list[dict] = []

    for N, grp in df_valid.groupby("N"):
        pearson_vals = grp["pearson_r"].dropna().values
        rmse_vals = grp["rmse"].dropna().values
        nrmse_vals = grp["nrmse"].dropna().values
        n_runs = len(grp)

        if len(pearson_vals) == 0:
            warnings.warn(f"H2 N={N}: no valid Pearson r values.")
            continue

        mean_r = float(np.mean(pearson_vals))
        std_r = float(np.std(pearson_vals, ddof=1)) if len(pearson_vals) > 1 else 0.0
        min_r = float(np.min(pearson_vals))
        max_r = float(np.max(pearson_vals))
        n_above_095 = int(np.sum(pearson_vals > 0.95))
        all_above_095 = bool(np.all(pearson_vals > 0.95))

        mean_rmse = float(np.mean(rmse_vals)) if len(rmse_vals) > 0 else float("nan")
        mean_nrmse = float(np.mean(nrmse_vals)) if len(nrmse_vals) > 0 else float("nan")

        mean_fr_ms = float(grp["fr_elapsed_ms"].mean())
        mean_lba_ms = float(grp["lba_elapsed_ms"].mean())
        mean_init_ms = float(grp["lba_init_ms"].mean()) if "lba_init_ms" in grp.columns else float("nan")
        mean_speedup = float(grp["speedup_ratio"].replace([np.inf], np.nan).mean())

        # Join h2_verdict from summary
        match = df_sum[df_sum["N"] == N]
        h2_verdict = match["h2_verdict"].iloc[0] if len(match) > 0 else "N/A"

        rows.append({
            "N": N,
            "n_runs": n_runs,
            "mean_pearson_r": mean_r,
            "std_pearson_r": std_r,
            "min_pearson_r": min_r,
            "max_pearson_r": max_r,
            "n_above_095": n_above_095,
            "all_above_095": all_above_095,
            "mean_rmse": mean_rmse,
            "mean_nrmse": mean_nrmse,
            "mean_fr_ms": mean_fr_ms,
            "mean_lba_ms": mean_lba_ms,
            "mean_init_ms": mean_init_ms,
            "mean_speedup": mean_speedup,
            "h2_verdict": h2_verdict,
        })

    df_out = pd.DataFrame(rows)

    # ---- Console table ----
    print("\n  H2 FORMAL STATISTICS")
    print("  " + "-" * 90)
    print(
        f"  {'N':>8} {'Runs':>5} {'Mean r':>8} {'Std r':>7} "
        f"{'Min r':>7} {'Max r':>7} {'n>0.95':>6} {'NRMSE':>8} "
        f"{'Speedup':>8} {'Verdict':>15}"
    )
    print("  " + "-" * 90)
    for _, r in df_out.iterrows():
        print(
            f"  {int(r['N']):>8,} {int(r['n_runs']):>5} "
            f"{r['mean_pearson_r']:>8.4f} {r['std_pearson_r']:>7.4f} "
            f"{r['min_pearson_r']:>7.4f} {r['max_pearson_r']:>7.4f} "
            f"{int(r['n_above_095']):>6} "
            f"{r['mean_nrmse']:>8.4f} "
            f"{r['mean_speedup']:>8.2f}x "
            f"{r['h2_verdict']:>15}"
        )
    print("  " + "-" * 90)

    # Per-run detail
    print("\n  Per-run Pearson r:")
    for N, grp in df_valid.groupby("N"):
        run_strs = []
        for _, row in grp.sort_values("run_idx").iterrows():
            r_val = row["pearson_r"]
            r_str = f"run{int(row['run_idx'])}={'N/A' if np.isnan(r_val) else f'{r_val:.4f}'}"
            run_strs.append(r_str)
        print(f"  N={int(N):>7,}:  {'  '.join(run_strs)}")

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved: {OUTPUT_PATH}")
    return df_out


def print_h2_verdict(df: pd.DataFrame) -> None:
    """Print overall H2 verdict.

    Parameters
    ----------
    df : pd.DataFrame
        Output of run_stats_h2().
    """
    if df.empty:
        print("\n  H2: Tidak cukup data untuk menentukan verdict.")
        return

    verdicts = df["h2_verdict"].tolist()
    all_r = df["min_pearson_r"].dropna().values

    r_min_global = float(np.min(all_r)) if len(all_r) > 0 else float("nan")
    r_max_global = df["max_pearson_r"].dropna().max() if len(df) > 0 else float("nan")

    print("\n  === H2 VERDICT ===")
    if all(v == "ACCEPTED" for v in verdicts):
        print(
            "  H2 DITERIMA: LBA dengan 5% landmark mencapai "
            "Pearson r > 0.95 pada semua ukuran graf yang diuji."
        )
    elif all(v in ("ACCEPTED", "ACCEPTED_PARTIAL") for v in verdicts):
        print(
            "  H2 DITERIMA SEBAGIAN: Pearson r > 0.90 pada semua N, "
            "namun min r <= 0.95 pada setidaknya satu ukuran graf. "
            "Diskusikan trade-off landmark fraction vs akurasi pada N besar."
        )
    else:
        print(
            "  H2 DITOLAK: min Pearson r <= 0.90 pada setidaknya satu "
            "ukuran graf N > 50,000. Pertimbangkan peningkatan landmark_fraction."
        )

    r_range_str = (
        f"{r_min_global:.4f} - {r_max_global:.4f}"
        if not np.isnan(r_min_global) else "N/A"
    )
    print(f"  Pearson r range across all runs: {r_range_str}")
