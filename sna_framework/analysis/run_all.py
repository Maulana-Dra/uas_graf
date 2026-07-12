"""
run_all.py
==========
Master runner for all SNA framework statistical analyses and figures.

Executes in order:
  1. H1 stats  → data/h1_stats_formal.csv
  2. H2 stats  → data/h2_stats_formal.csv
  3. H4 stats  → data/h4_stats_formal.csv
  4. H1 figures → figures/h1_*.png  (4 files)
  5. H2 figures → figures/h2_*.png  (3 files)
  6. H4 figures → figures/h4_*.png  (4 files)

Usage:
  python analysis/run_all.py          # from sna_framework/
  cd analysis && python run_all.py    # from analysis/
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — importable from sna_framework/ or analysis/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports — analysis package
# ---------------------------------------------------------------------------
from analysis.stats_h1 import run_stats_h1, print_h1_verdict
from analysis.stats_h2 import run_stats_h2, print_h2_verdict
from analysis.stats_h4 import run_stats_h4, print_h4_verdict
from analysis.visualize_h1 import plot_h1_all
from analysis.visualize_h2 import plot_h2_all
from analysis.visualize_h4 import plot_h4_all

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA = _ROOT / "data"
FIGS = _ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def _section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def main() -> None:
    """Run all analyses and generate all figures."""
    _section("SNA FRAMEWORK — LANGKAH 5: ANALISIS STATISTIK & VISUALISASI")

    errors: list[str] = []

    # -----------------------------------------------------------------------
    # H1 Stats
    # -----------------------------------------------------------------------
    _section("H1 STATS")
    df_h1 = None
    try:
        df_h1 = run_stats_h1(DATA / "h1_raw.csv", DATA / "h1_summary.csv")
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        errors.append(f"H1 stats: {exc}")
    except Exception as exc:
        print(f"  [ERROR] H1 stats failed: {exc}")
        traceback.print_exc()
        errors.append(f"H1 stats: {exc}")

    if df_h1 is not None:
        print_h1_verdict(df_h1)

    # -----------------------------------------------------------------------
    # H2 Stats
    # -----------------------------------------------------------------------
    _section("H2 STATS")
    df_h2 = None
    try:
        df_h2 = run_stats_h2(DATA / "h2_raw.csv", DATA / "h2_summary.csv")
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        errors.append(f"H2 stats: {exc}")
    except Exception as exc:
        print(f"  [ERROR] H2 stats failed: {exc}")
        traceback.print_exc()
        errors.append(f"H2 stats: {exc}")

    if df_h2 is not None:
        print_h2_verdict(df_h2)

    # -----------------------------------------------------------------------
    # H4 Stats
    # -----------------------------------------------------------------------
    _section("H4 STATS")
    df_h4 = None
    try:
        df_h4 = run_stats_h4(DATA / "h4_raw.csv", DATA / "h4_summary.csv")
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        errors.append(f"H4 stats: {exc}")
    except Exception as exc:
        print(f"  [ERROR] H4 stats failed: {exc}")
        traceback.print_exc()
        errors.append(f"H4 stats: {exc}")

    if df_h4 is not None:
        print_h4_verdict(df_h4)

    # -----------------------------------------------------------------------
    # H1 Figures
    # -----------------------------------------------------------------------
    _section("GENERATING H1 FIGURES")
    try:
        plot_h1_all(DATA / "h1_raw.csv", DATA / "h1_summary.csv", FIGS)
        print("  H1 figures: done")
    except Exception as exc:
        print(f"  [ERROR] H1 figures: {exc}")
        traceback.print_exc()
        errors.append(f"H1 figures: {exc}")

    # -----------------------------------------------------------------------
    # H2 Figures
    # -----------------------------------------------------------------------
    _section("GENERATING H2 FIGURES")
    try:
        plot_h2_all(DATA / "h2_raw.csv", DATA / "h2_summary.csv", FIGS)
        print("  H2 figures: done")
    except Exception as exc:
        print(f"  [ERROR] H2 figures: {exc}")
        traceback.print_exc()
        errors.append(f"H2 figures: {exc}")

    # -----------------------------------------------------------------------
    # H4 Figures
    # -----------------------------------------------------------------------
    _section("GENERATING H4 FIGURES")
    try:
        plot_h4_all(DATA / "h4_raw.csv", DATA / "h4_summary.csv", FIGS)
        print("  H4 figures: done")
    except Exception as exc:
        print(f"  [ERROR] H4 figures: {exc}")
        traceback.print_exc()
        errors.append(f"H4 figures: {exc}")

    # -----------------------------------------------------------------------
    # Final verdict summary
    # -----------------------------------------------------------------------
    _section("FINAL VERDICT SUMMARY")
    if df_h1 is not None:
        print_h1_verdict(df_h1)
    else:
        print("  H1: Data belum tersedia — jalankan experiments/run_h1.py")

    if df_h2 is not None:
        print_h2_verdict(df_h2)
    else:
        print("  H2: Data belum tersedia — jalankan experiments/run_h2.py")

    if df_h4 is not None:
        print_h4_verdict(df_h4)
    else:
        print("  H4: Data belum tersedia — jalankan experiments/run_h4.py")

    # -----------------------------------------------------------------------
    # File inventory
    # -----------------------------------------------------------------------
    _section("OUTPUT FILES")
    print("\n  Formal stats CSVs:")
    for fname in ["h1_stats_formal.csv", "h2_stats_formal.csv",
                  "h4_stats_formal.csv"]:
        p = DATA / fname
        status = f"{p.stat().st_size:,} bytes" if p.exists() else "NOT CREATED"
        print(f"    {fname:30s}  {status}")

    print("\n  Figures:")
    expected_figs = [
        "h1_efficiency_by_churn.png",
        "h1_speedup_boxplot.png",
        "h1_time_comparison.png",
        "h1_affected_nodes.png",
        "h2_pearson_scatter.png",
        "h2_pearson_by_N.png",
        "h2_speed_comparison.png",
        "h4_steps_boxplot.png",
        "h4_mean_steps_heatmap.png",
        "h4_closeness_faster_pct.png",
        "h4_pvalue_significance.png",
    ]
    n_ok = 0
    for fname in expected_figs:
        p = FIGS / fname
        if p.exists():
            status = f"{p.stat().st_size:,} bytes  [OK]"
            n_ok += 1
        else:
            status = "NOT CREATED  [!!]"
        print(f"    {fname:35s}  {status}")

    print(f"\n  {n_ok}/{len(expected_figs)} figures created.")

    if errors:
        _section("ERRORS / WARNINGS")
        for e in errors:
            print(f"  [!] {e}")

    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
