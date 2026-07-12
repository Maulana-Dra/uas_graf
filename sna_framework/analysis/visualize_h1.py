"""
visualize_h1.py
===============
Publication-ready figures for H1: ICC vs Full Recompute speed comparison.

Generates 4 figures into figures/:
  h1_efficiency_by_churn.png  – grouped bar chart of mean efficiency%
  h1_speedup_boxplot.png      – box plots of speedup_ratio (N x churn grid)
  h1_time_comparison.png      – line chart FR vs ICC over batches (churn=5%)
  h1_affected_nodes.png       – n_affected per batch per churn rate
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
STYLE: dict = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "patch.edgecolor": "white",
}
mpl.rcParams.update(STYLE)

# Color palette
COLOR_FR     = "#2166ac"
COLOR_ICC    = "#d6604d"
COLOR_LBA    = "#4dac26"
COLOR_CL     = "#7b2d8b"
COLOR_DEG    = "#e08214"
COLOR_TARGET = "#bababa"

# ICC churn-rate gradient (light → dark red-orange)
CHURN_COLORS = ["#f4a582", "#d6604d", "#8b1a1a"]


def _placeholder(out_path: Path, message: str) -> None:
    """Save a blank figure with an error/info message."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", fontsize=12,
        transform=ax.transAxes, wrap=True,
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#ffc107")
    )
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  [PLACEHOLDER] {out_path.name}: {message}")


def _load_and_check(
    raw_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load CSVs; return None and warn if missing."""
    if not raw_path.exists():
        warnings.warn(f"H1 raw CSV not found: {raw_path}")
        return None
    if not summary_path.exists():
        warnings.warn(f"H1 summary CSV not found: {summary_path}")
        return None
    return pd.read_csv(raw_path), pd.read_csv(summary_path)


# ---------------------------------------------------------------------------
# Figure 1 – Efficiency by churn
# ---------------------------------------------------------------------------

def _fig_efficiency_by_churn(
    df_sum: pd.DataFrame, out_path: Path
) -> None:
    """Grouped bar chart of mean_efficiency_pct by N and churn_rate."""
    if df_sum.empty or "mean_efficiency_pct" not in df_sum.columns:
        _placeholder(out_path, "H1 summary data insufficient for efficiency chart.")
        return

    N_vals = sorted(df_sum["N"].unique())
    churn_vals = sorted(df_sum["churn_rate"].unique())
    n_groups = len(N_vals)
    n_bars = len(churn_vals)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_groups)
    total_width = 0.7
    bar_width = total_width / max(n_bars, 1)

    for ci, (churn, color) in enumerate(zip(churn_vals, CHURN_COLORS)):
        offsets = x + (ci - n_bars / 2 + 0.5) * bar_width
        sub = df_sum[df_sum["churn_rate"] == churn].set_index("N")
        means = [sub.loc[N, "mean_efficiency_pct"] if N in sub.index else np.nan for N in N_vals]
        stds  = [sub.loc[N, "std_efficiency_pct"]  if N in sub.index else 0.0    for N in N_vals]

        bars = ax.bar(
            offsets, means, width=bar_width * 0.9,
            color=color, label=f"Churn {churn*100:.0f}%",
            yerr=stds, capsize=4, error_kw={"elinewidth": 1.5},
            alpha=0.88,
        )
        # Annotate bar values
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.1 + 1,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold"
                )

    # Target line
    ax.axhline(80, color=COLOR_TARGET, linestyle="--", linewidth=1.8,
               label="Target 80%", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={int(n):,}" for n in N_vals], fontsize=11)
    ax.set_xlabel("Ukuran Graf (N)")
    ax.set_ylabel("Mean Efficiency ICC vs FR (%)")
    ax.set_title("H1: Efisiensi ICC vs Full Recompute per Ukuran Graf")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=min(0, df_sum["mean_efficiency_pct"].min() - 10))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2 – Speedup box plot grid
# ---------------------------------------------------------------------------

def _fig_speedup_boxplot(
    df_raw: pd.DataFrame, out_path: Path
) -> None:
    """3×3 box plot grid of speedup_ratio, by N (row) and churn_rate (col)."""
    if df_raw.empty or "speedup_ratio" not in df_raw.columns:
        _placeholder(out_path, "H1 raw data insufficient for speedup boxplot.")
        return

    N_vals = sorted(df_raw["N"].unique())
    churn_vals = sorted(df_raw["churn_rate"].unique())
    nrows, ncols = max(len(N_vals), 1), max(len(churn_vals), 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), squeeze=False)
    fig.suptitle(
        "H1: Distribusi Speedup Ratio per Kombinasi (N x Churn)",
        fontsize=14, fontweight="bold", y=1.01
    )

    for ri, N in enumerate(N_vals):
        for ci, churn in enumerate(churn_vals):
            ax = axes[ri][ci]
            sub = df_raw[
                (df_raw["N"] == N) & (df_raw["churn_rate"] == churn)
            ]["speedup_ratio"].replace([np.inf, -np.inf], np.nan).dropna()

            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
            else:
                bp = ax.boxplot(
                    sub, patch_artist=True,
                    boxprops=dict(facecolor=CHURN_COLORS[ci], alpha=0.7),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                )

            # Reference lines
            ax.axhline(1, color=COLOR_TARGET, linewidth=1.4, linestyle="-",
                       label="No speedup (1×)")
            ax.axhline(5, color="#666666", linewidth=1.2, linestyle="--",
                       label="5× speedup")

            ax.set_title(f"N={int(N):,}  Churn={churn*100:.0f}%", fontsize=10)
            ax.set_xlabel("Churn Rate", fontsize=9)
            ax.set_ylabel("Speedup Ratio" if ci == 0 else "", fontsize=9)
            ax.set_xticks([])

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3 – Time comparison line chart (churn=5%)
# ---------------------------------------------------------------------------

def _fig_time_comparison(
    df_raw: pd.DataFrame, out_path: Path,
    target_churn: float = 0.05
) -> None:
    """Line chart: FR vs ICC elapsed time per batch at churn=5%."""
    sub = df_raw[df_raw["churn_rate"] == target_churn]
    if sub.empty:
        # Fallback: use any available churn
        if df_raw.empty:
            _placeholder(out_path, "H1 raw data missing for time comparison.")
            return
        target_churn = df_raw["churn_rate"].iloc[0]
        sub = df_raw[df_raw["churn_rate"] == target_churn]

    N_vals = sorted(sub["N"].unique())
    n_panels = max(len(N_vals), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    fig.suptitle(
        f"H1: Tren Waktu Komputasi per Batch (Churn {target_churn*100:.0f}%)",
        fontsize=14, fontweight="bold"
    )

    for pi, N in enumerate(N_vals):
        ax = axes[0][pi]
        grp = sub[sub["N"] == N].sort_values("batch_index")

        if len(grp) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        batches = grp["batch_index"].values
        fr_ms = grp["fr_elapsed_ms"].values
        icc_ms = grp["icc_elapsed_ms"].values

        # Line + shading (use ±1 stddev from rolling window if n>=5, else raw values)
        ax.plot(batches, fr_ms, color=COLOR_FR, linewidth=2,
                label="Full Recompute", marker="o", markersize=3)
        ax.plot(batches, icc_ms, color=COLOR_ICC, linewidth=2,
                label="ICC", marker="s", markersize=3)

        if len(batches) >= 3:
            w = max(1, len(batches) // 5)
            fr_roll_std = pd.Series(fr_ms).rolling(w, min_periods=1).std().fillna(0).values
            icc_roll_std = pd.Series(icc_ms).rolling(w, min_periods=1).std().fillna(0).values
            ax.fill_between(batches, fr_ms - fr_roll_std, fr_ms + fr_roll_std,
                            color=COLOR_FR, alpha=0.18)
            ax.fill_between(batches, icc_ms - icc_roll_std, icc_ms + icc_roll_std,
                            color=COLOR_ICC, alpha=0.18)

        ax.set_yscale("log")
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("Elapsed Time (ms, log scale)" if pi == 0 else "")
        ax.set_title(f"N = {int(N):,}, Churn = {target_churn*100:.0f}%")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 4 – Affected nodes
# ---------------------------------------------------------------------------

def _fig_affected_nodes(
    df_raw: pd.DataFrame, out_path: Path
) -> None:
    """Line chart: n_affected per batch, one line per churn_rate, panels per N."""
    if df_raw.empty or "n_affected" not in df_raw.columns:
        _placeholder(out_path, "H1 raw data missing n_affected column.")
        return

    N_vals = sorted(df_raw["N"].unique())
    churn_vals = sorted(df_raw["churn_rate"].unique())
    n_panels = max(len(N_vals), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    fig.suptitle(
        "H1: Jumlah Node yang Di-rekomputasi ICC per Batch",
        fontsize=14, fontweight="bold"
    )

    for pi, N in enumerate(N_vals):
        ax = axes[0][pi]
        sub_N = df_raw[df_raw["N"] == N]

        for ci, (churn, color) in enumerate(zip(churn_vals, CHURN_COLORS)):
            sub = sub_N[sub_N["churn_rate"] == churn].sort_values("batch_index")
            if len(sub) == 0:
                continue
            ax.plot(
                sub["batch_index"].values,
                sub["n_affected"].values,
                color=color, linewidth=2, marker="o", markersize=3,
                label=f"Churn {churn*100:.0f}%"
            )

        ax.set_xlabel("Batch Index")
        ax.set_ylabel("n_affected (nodes)" if pi == 0 else "")
        ax.set_title(f"N = {int(N):,}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_h1_all(
    raw_path: Path | str,
    summary_path: Path | str,
    out_dir: Path | str,
) -> None:
    """Generate all 4 H1 figures and save to out_dir.

    Parameters
    ----------
    raw_path : Path or str
    summary_path : Path or str
    out_dir : Path or str
        Directory where PNG files are saved.
    """
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _load_and_check(raw_path, summary_path)
    if result is None:
        for name in [
            "h1_efficiency_by_churn.png", "h1_speedup_boxplot.png",
            "h1_time_comparison.png", "h1_affected_nodes.png",
        ]:
            _placeholder(out_dir / name, f"Data missing: {raw_path.name}")
        return

    df_raw, df_sum = result

    for fig_name, func, kwargs in [
        ("h1_efficiency_by_churn.png", _fig_efficiency_by_churn,
         {"df_sum": df_sum}),
        ("h1_speedup_boxplot.png", _fig_speedup_boxplot,
         {"df_raw": df_raw}),
        ("h1_time_comparison.png", _fig_time_comparison,
         {"df_raw": df_raw}),
        ("h1_affected_nodes.png", _fig_affected_nodes,
         {"df_raw": df_raw}),
    ]:
        try:
            func(out_path=out_dir / fig_name, **kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            _placeholder(out_dir / fig_name, f"Error: {exc}\n{tb[:300]}")
        finally:
            plt.close("all")
