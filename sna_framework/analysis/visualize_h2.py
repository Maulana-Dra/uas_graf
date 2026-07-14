"""
visualize_h2.py
===============
Publication-ready figures for H2: LBA accuracy vs Full Recompute.

Generates 3 figures into figures/:
  h2_pearson_scatter.png   – scatter plot FR vs LBA centrality scores
  h2_pearson_by_N.png      – bar + dot chart of Pearson r per N
  h2_speed_comparison.png  – grouped bars FR / LBA / LBA-init timing
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
# Global style (must match across all visualize_*.py)
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

COLOR_FR     = "#2166ac"
COLOR_ICC    = "#d6604d"
COLOR_LBA    = "#4dac26"
COLOR_LBA_DK = "#2a6114"   # darker green for LBA init
COLOR_CL     = "#7b2d8b"
COLOR_DEG    = "#e08214"
COLOR_TARGET = "#bababa"

# N axis labels
_N_LABEL = {50_000: "50k", 75_000: "75k", 100_000: "100k",
            1_000: "1k", 5_000: "5k", 10_000: "10k"}


def _placeholder(out_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", fontsize=11,
        transform=ax.transAxes, wrap=True,
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#ffc107")
    )
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  [PLACEHOLDER] {out_path.name}: {message}")


# ---------------------------------------------------------------------------
# Figure 1 – Scatter: FR vs LBA per-node centrality
# ---------------------------------------------------------------------------

def _fig_pearson_scatter(out_path: Path, n_scatter: int = 2_000) -> None:
    """Re-run engine to get per-node scores, then scatter FR vs LBA."""
    try:
        from engine.graph_generator import GraphGenerator
        from engine.full_recompute import FullRecompute
        from engine.lba import LandmarkApproximation
        from scipy.stats import pearsonr
    except ImportError as exc:
        _placeholder(
            out_path,
            "Engine import failed — run experiments/run_h2.py first.\n"
            f"Error: {exc}"
        )
        return

    # Use N=50,000 as representative (matching the first scale of the H2 methodology)
    N_REP = 50_000
    SEED = 42
    print(
        f"  Generating representative scatter (N={N_REP:,}, seed={SEED})...\n"
        "  This may take a few minutes..."
    )

    try:
        gen = GraphGenerator()
        G = gen.generate_ba_graph(N=N_REP, m=2, seed=SEED)

        fr_engine = FullRecompute()
        fr_result = fr_engine.compute(G)
        fr_cent = fr_result["centrality"]

        lba = LandmarkApproximation(G, landmark_fraction=0.05, seed=SEED)
        lba_result = lba.compute_approximation()
        lba_cent = lba_result["centrality"]
        n_landmarks = lba_result["n_landmarks"]

        nodes = sorted(G.nodes())
        fr_vals = np.array([fr_cent[n] for n in nodes])
        lba_vals = np.array([lba_cent[n] for n in nodes])

        r, _ = pearsonr(fr_vals, lba_vals)

        # Sample for density scatter
        rng = np.random.default_rng(0)
        idx = rng.choice(len(nodes), size=min(n_scatter, len(nodes)), replace=False)
        fr_samp = fr_vals[idx]
        lba_samp = lba_vals[idx]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            fr_samp, lba_samp,
            alpha=0.35, s=12, color=COLOR_LBA, edgecolors="none"
        )
        # Diagonal reference y=x
        mn = min(fr_vals.min(), lba_vals.min())
        mx = max(fr_vals.max(), lba_vals.max())
        ax.plot([mn, mx], [mn, mx], color=COLOR_TARGET,
                linewidth=1.5, linestyle="--", label="y = x (perfect)")

        ax.text(
            0.05, 0.93,
            f"Pearson r = {r:.4f}\nN = {N_REP:,}\nLandmarks = {n_landmarks} (5%)",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85)
        )
        ax.set_xlabel("Closeness Centrality (Full Recompute)")
        ax.set_ylabel("Closeness Centrality (LBA, 5% Landmark)")
        ax.set_title(
            f"H2: Korelasi Skor LBA vs Full Recompute "
            f"(N={N_REP:,}, sampel {n_scatter:,} node)"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close("all")
        print(f"  Saved: {out_path.name}")

    except Exception as exc:
        tb = traceback.format_exc()
        _placeholder(
            out_path,
            f"Scatter plot error (N={N_REP}):\n{exc}\n\n{tb[:400]}"
        )


# ---------------------------------------------------------------------------
# Figure 2 – Pearson r by N (bar + dot overlay)
# ---------------------------------------------------------------------------

def _fig_pearson_by_N(df_raw: pd.DataFrame, df_sum: pd.DataFrame,
                      out_path: Path) -> None:
    """Bar chart (mean r ± std) + dot overlay (individual runs) per N."""
    valid = df_raw[df_raw["h2_criterion"].isin(["MEETS_TARGET", "BELOW_TARGET"])]
    if valid.empty:
        _placeholder(out_path, "H2: no valid runs for Pearson r chart.")
        return

    N_vals = sorted(valid["N"].unique())
    x_pos = np.arange(len(N_vals))
    xlabels = [_N_LABEL.get(int(N), f"{int(N):,}") for N in N_vals]

    fig, ax = plt.subplots(figsize=(9, 6))

    means, stds, all_r_lists = [], [], []
    for N in N_vals:
        sub = valid[valid["N"] == N]["pearson_r"].dropna().values
        means.append(sub.mean() if len(sub) > 0 else np.nan)
        stds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
        all_r_lists.append(sub)

    bars = ax.bar(
        x_pos, means, width=0.5, color=COLOR_LBA, alpha=0.8,
        yerr=stds, capsize=6, error_kw={"elinewidth": 2},
        label="Mean Pearson r"
    )

    # Dot overlay — jittered
    rng = np.random.default_rng(1)
    for xi, r_vals in zip(x_pos, all_r_lists):
        jitter = rng.uniform(-0.08, 0.08, size=len(r_vals))
        ax.scatter(
            np.full(len(r_vals), xi) + jitter,
            r_vals, color=COLOR_LBA_DK, s=50, alpha=0.75, zorder=5
        )

    # Bar annotations
    for bar, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                m + s + 0.003,
                f"{m:.2f}±{s:.2f}",
                ha="center", va="bottom", fontsize=9.5
            )

    # Target line
    ax.axhline(0.95, color=COLOR_TARGET, linestyle="--", linewidth=2,
               label="Target r > 0.95")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_xlabel("Ukuran Graf (N)")
    ax.set_ylabel("Pearson r")
    ax.set_title("H2: Pearson r per Ukuran Graf (5 runs per N)")

    # Zoom Y-axis: between 0.85 and 1
    all_r = np.concatenate(all_r_lists)
    ymin = max(0.80, np.nanmin(all_r) - 0.05) if len(all_r) > 0 else 0.85
    ax.set_ylim(ymin, 1.01)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3 – Speed comparison
# ---------------------------------------------------------------------------

def _fig_speed_comparison(df_raw: pd.DataFrame, df_sum: pd.DataFrame,
                           out_path: Path) -> None:
    """Grouped bar chart: FR vs LBA-compute vs LBA-init time per N (log scale)."""
    valid = df_raw[df_raw["h2_criterion"].isin(["MEETS_TARGET", "BELOW_TARGET"])]
    if valid.empty:
        _placeholder(out_path, "H2: no valid runs for speed comparison.")
        return

    N_vals = sorted(valid["N"].unique())
    x_pos = np.arange(len(N_vals))
    xlabels = [_N_LABEL.get(int(N), f"{int(N):,}") for N in N_vals]

    fr_means, lba_means, init_means, speedup_means = [], [], [], []
    for N in N_vals:
        sub = valid[valid["N"] == N]
        fr_means.append(sub["fr_elapsed_ms"].mean())
        lba_means.append(sub["lba_elapsed_ms"].mean())
        init_means.append(sub["lba_init_ms"].mean() if "lba_init_ms" in sub.columns else np.nan)
        sr = sub["speedup_ratio"].replace([np.inf], np.nan).mean()
        speedup_means.append(sr)

    bar_w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    b_fr   = ax.bar(x_pos - bar_w, fr_means,   width=bar_w, color=COLOR_FR,
                    label="FR elapsed", alpha=0.85)
    b_lba  = ax.bar(x_pos,          lba_means,  width=bar_w, color=COLOR_LBA,
                    label="LBA compute", alpha=0.85)
    b_init = ax.bar(x_pos + bar_w,  init_means, width=bar_w, color=COLOR_LBA_DK,
                    label="LBA init (BFS)", alpha=0.85)

    # Annotate speedup above LBA bars
    for bar, sp in zip(b_lba, speedup_means):
        if not np.isnan(sp):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.5,
                f"{sp:.1f}x\nfaster",
                ha="center", va="bottom", fontsize=9,
                color=COLOR_LBA_DK, fontweight="bold"
            )

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_xlabel("Ukuran Graf (N)")
    ax.set_ylabel("Waktu (ms, skala log)")
    ax.set_title("H2: Perbandingan Waktu FR vs LBA (termasuk Init Landmark)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_h2_all(
    raw_path: Path | str,
    summary_path: Path | str,
    out_dir: Path | str,
) -> None:
    """Generate all 3 H2 figures and save to out_dir.

    Parameters
    ----------
    raw_path : Path or str
    summary_path : Path or str
    out_dir : Path or str
    """
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    df_sum = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    # Figure 1: scatter — requires re-running engine
    try:
        _fig_pearson_scatter(out_path=out_dir / "h2_pearson_scatter.png")
    except Exception as exc:
        _placeholder(out_dir / "h2_pearson_scatter.png", f"Scatter error: {exc}")
    finally:
        plt.close("all")

    # Figure 2: Pearson r bar + dots
    try:
        _fig_pearson_by_N(df_raw, df_sum, out_dir / "h2_pearson_by_N.png")
    except Exception as exc:
        _placeholder(out_dir / "h2_pearson_by_N.png", f"Error: {exc}")
    finally:
        plt.close("all")

    # Figure 3: speed comparison
    try:
        _fig_speed_comparison(df_raw, df_sum, out_dir / "h2_speed_comparison.png")
    except Exception as exc:
        _placeholder(out_dir / "h2_speed_comparison.png", f"Error: {exc}")
    finally:
        plt.close("all")
