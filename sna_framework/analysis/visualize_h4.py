"""
visualize_h4.py
===============
Publication-ready figures for H4: Closeness vs Degree seed in disaster scenario.

Generates 4 figures into figures/:
  h4_steps_boxplot.png          – 3×3 side-by-side box plots
  h4_mean_steps_heatmap.png     – dual heatmaps (closeness | degree)
  h4_closeness_faster_pct.png   – heatmap of % runs where closeness wins
  h4_pvalue_significance.png    – heatmap of statistical significance
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

COLOR_FR     = "#2166ac"
COLOR_ICC    = "#d6604d"
COLOR_LBA    = "#4dac26"
COLOR_CL     = "#7b2d8b"
COLOR_DEG    = "#e08214"
COLOR_TARGET = "#bababa"

CENSORED_VALUE = 501


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


def _load(raw_path: Path, summary_path: Path
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    df_sum = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    return df_raw, df_sum


# ---------------------------------------------------------------------------
# Figure 1 – 3×3 box plot grid
# ---------------------------------------------------------------------------

def _fig_steps_boxplot(df_raw: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side box plots of steps_closeness vs steps_degree, 3×3 grid."""
    if df_raw.empty:
        _placeholder(out_path, "H4 raw data missing for boxplot.")
        return

    N_vals = sorted(df_raw["N"].unique())
    fr_vals = sorted(df_raw["failure_rate"].unique())
    nrows, ncols = max(len(N_vals), 1), max(len(fr_vals), 1)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(16, 12), squeeze=False,
        sharey=False
    )
    fig.suptitle(
        "H4: Distribusi Steps to Coverage (Closeness vs Degree Seed)",
        fontsize=14, fontweight="bold"
    )

    for ri, N in enumerate(N_vals):
        for ci, fr in enumerate(fr_vals):
            ax = axes[ri][ci]
            sub = df_raw[(df_raw["N"] == N) & (df_raw["failure_rate"] == fr)]

            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f"N={int(N):,}, Fail={fr*100:.0f}%", fontsize=10)
                continue

            data_c = sub["steps_closeness"].values.tolist()
            data_d = sub["steps_degree"].values.tolist()

            bp = ax.boxplot(
                [data_c, data_d],
                patch_artist=True,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
            )
            ax.set_xticklabels(["Closeness", "Degree"])
            bp["boxes"][0].set_facecolor(COLOR_CL)
            bp["boxes"][0].set_alpha(0.7)
            if len(bp["boxes"]) > 1:
                bp["boxes"][1].set_facecolor(COLOR_DEG)
                bp["boxes"][1].set_alpha(0.7)

            ax.axhline(
                CENSORED_VALUE, color=COLOR_TARGET, linestyle="--",
                linewidth=1.5, label="Censored (501)"
            )
            ax.set_title(f"N={int(N):,}, Fail={fr*100:.0f}%", fontsize=10)
            ax.set_ylabel("Steps to 50% Coverage" if ci == 0 else "", fontsize=9)
            if ri == 0 and ci == ncols - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2 – Dual heatmaps: mean steps closeness | degree
# ---------------------------------------------------------------------------

def _fig_mean_steps_heatmap(df_sum: pd.DataFrame, out_path: Path) -> None:
    """Two side-by-side heatmaps: mean_steps_closeness and mean_steps_degree."""
    if df_sum.empty:
        _placeholder(out_path, "H4 summary data missing for heatmap.")
        return

    N_vals = sorted(df_sum["N"].unique())
    fr_vals = sorted(df_sum["failure_rate"].unique())

    def _build_matrix(col: str) -> np.ndarray:
        mat = np.full((len(N_vals), len(fr_vals)), np.nan)
        for ri, N in enumerate(N_vals):
            for ci, fr in enumerate(fr_vals):
                sub = df_sum[(df_sum["N"] == N) & (df_sum["failure_rate"] == fr)]
                if len(sub) > 0 and not pd.isna(sub[col].iloc[0]):
                    mat[ri, ci] = float(sub[col].iloc[0])
        return mat

    mat_c = _build_matrix("mean_steps_closeness")
    mat_d = _build_matrix("mean_steps_degree")

    # Shared color scale
    vmin = np.nanmin([mat_c, mat_d])
    vmax = np.nanmax([mat_c, mat_d])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("H4: Rata-rata Steps to 50% Coverage", fontsize=14,
                 fontweight="bold")

    col_labels = [f"{int(fr*100)}%" for fr in fr_vals]
    row_labels = [f"N={int(N):,}" for N in N_vals]

    for ax, mat, title in zip(
        axes,
        [mat_c, mat_d],
        ["Mean Steps (Closeness Seed)", "Mean Steps (Degree Seed)"]
    ):
        im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(fr_vals)))
        ax.set_xticklabels(col_labels, fontsize=11)
        ax.set_yticks(range(len(N_vals)))
        ax.set_yticklabels(row_labels, fontsize=11)
        ax.set_xlabel("Failure Rate")
        ax.set_title(title, fontsize=12)
        ax.grid(False)

        # Annotate cells
        for ri in range(len(N_vals)):
            for ci in range(len(fr_vals)):
                val = mat[ri, ci]
                txt = f"{val:.1f}" if not np.isnan(val) else "N/A"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val > (vmin + vmax) / 1.5 else "black")

    plt.colorbar(im, ax=axes, label="Steps to Coverage", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3 – Closeness faster % heatmap
# ---------------------------------------------------------------------------

def _fig_closeness_faster_pct(df_sum: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of pct_closeness_faster, centered at 50 (neutral)."""
    if df_sum.empty or "pct_closeness_faster" not in df_sum.columns:
        _placeholder(out_path, "H4: pct_closeness_faster data missing.")
        return

    N_vals = sorted(df_sum["N"].unique())
    fr_vals = sorted(df_sum["failure_rate"].unique())

    mat = np.full((len(N_vals), len(fr_vals)), np.nan)
    for ri, N in enumerate(N_vals):
        for ci, fr in enumerate(fr_vals):
            sub = df_sum[(df_sum["N"] == N) & (df_sum["failure_rate"] == fr)]
            if len(sub) > 0:
                v = sub["pct_closeness_faster"].iloc[0]
                if not pd.isna(v):
                    mat[ri, ci] = float(v)

    fig, ax = plt.subplots(figsize=(9, 5))

    im = ax.imshow(
        mat, cmap="RdYlGn", vmin=0, vmax=100,
        aspect="auto"
    )

    col_labels = [f"Fail {int(fr*100)}%" for fr in fr_vals]
    row_labels = [f"N={int(N):,}" for N in N_vals]
    ax.set_xticks(range(len(fr_vals)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(N_vals)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Failure Rate")
    ax.set_title("H4: Persentase Run di mana Closeness Seed Lebih Cepat",
                 fontsize=12)
    ax.grid(False)

    for ri in range(len(N_vals)):
        for ci in range(len(fr_vals)):
            val = mat[ri, ci]
            txt = f"{val:.1f}%" if not np.isnan(val) else "N/A"
            color = "white" if (not np.isnan(val) and (val > 75 or val < 25)) else "black"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, label="% Runs Where Closeness Faster",
                        shrink=0.85)
    cbar.ax.axhline(50, color="black", linewidth=1.5, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 4 – p-value significance heatmap
# ---------------------------------------------------------------------------

def _fig_pvalue_significance(df_sum: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of p-values with significance annotation."""
    if df_sum.empty or "p_value" not in df_sum.columns:
        _placeholder(out_path, "H4: p_value data missing.")
        return

    N_vals = sorted(df_sum["N"].unique())
    fr_vals = sorted(df_sum["failure_rate"].unique())

    pmat = np.full((len(N_vals), len(fr_vals)), np.nan)
    verdict_mat: list[list[str]] = [["" for _ in fr_vals] for _ in N_vals]

    for ri, N in enumerate(N_vals):
        for ci, fr in enumerate(fr_vals):
            sub = df_sum[(df_sum["N"] == N) & (df_sum["failure_rate"] == fr)]
            if len(sub) > 0:
                p = sub["p_value"].iloc[0]
                if not pd.isna(p):
                    pmat[ri, ci] = float(p)
                    verdict_mat[ri][ci] = (
                        "sig" if float(p) < 0.05 else "n.s."
                    )

    fig, ax = plt.subplots(figsize=(9, 5))

    # RdYlGn_r: green = low p (significant), red = high p
    im = ax.imshow(
        pmat, cmap="RdYlGn_r", vmin=0, vmax=1,
        aspect="auto"
    )

    col_labels = [f"Fail {int(fr*100)}%" for fr in fr_vals]
    row_labels = [f"N={int(N):,}" for N in N_vals]
    ax.set_xticks(range(len(fr_vals)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(N_vals)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title("H4: Signifikansi Statistik per Kombinasi (alpha=0.05)",
                 fontsize=12)
    ax.grid(False)

    for ri in range(len(N_vals)):
        for ci in range(len(fr_vals)):
            val = pmat[ri, ci]
            verd = verdict_mat[ri][ci]
            if not np.isnan(val):
                mark = "v sig" if verd == "sig" else "x n.s."
                label = f"p={val:.3f}\n{mark}"
            else:
                label = "N/A"
            color = "white" if (not np.isnan(val) and val < 0.15) else "black"
            ax.text(ci, ri, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="p-value", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_h4_all(
    raw_path: Path | str,
    summary_path: Path | str,
    out_dir: Path | str,
) -> None:
    """Generate all 4 H4 figures and save to out_dir.

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

    df_raw, df_sum = _load(raw_path, summary_path)

    for fig_name, func, kwargs in [
        ("h4_steps_boxplot.png", _fig_steps_boxplot,
         {"df_raw": df_raw}),
        ("h4_mean_steps_heatmap.png", _fig_mean_steps_heatmap,
         {"df_sum": df_sum}),
        ("h4_closeness_faster_pct.png", _fig_closeness_faster_pct,
         {"df_sum": df_sum}),
        ("h4_pvalue_significance.png", _fig_pvalue_significance,
         {"df_sum": df_sum}),
    ]:
        try:
            func(out_path=out_dir / fig_name, **kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            _placeholder(out_dir / fig_name, f"Error: {exc}\n{tb[:300]}")
        finally:
            plt.close("all")
