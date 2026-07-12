"""
run_h2.py
=========
H2 Experiment: LBA accuracy vs Full Recompute on large BA graphs.

Hypothesis H2
-------------
Landmark-Based Approximation (LBA) with 5% landmarks produces
Pearson r > 0.95 versus exact closeness centrality (Full Recompute)
on Barabasi-Albert graphs with N > 50,000 nodes.

This is a goodness-of-fit / correlation check — NOT a hypothesis test
with p-value. Accept if Pearson r > 0.95 across all tested graph sizes.

Metrics reported per run
------------------------
  pearson_r     : Pearson correlation between LBA and FR centrality vectors
  rmse          : Root Mean Square Error (absolute error in centrality units)
  nrmse         : RMSE normalized by FR score range (dimensionless, [0,1])
  speedup_ratio : fr_elapsed_ms / lba_elapsed_ms (LBA compute pass only)
  lba_init_ms   : BFS precomputation cost inside LandmarkApproximation.__init__
                  (logged for transparency but excluded from speedup_ratio)

Experimental design
-------------------
  Graph sizes   : 50,000 / 75,000 / 100,000 nodes
  Landmark frac : 5% of N
  Repetitions   : 5 independent runs per N (seed = 42 + run_idx)
  BA params     : m=2 (avg degree ~4)

Output files
------------
  data/h2_raw.csv     – 15 rows  (5 runs x 3 sizes)
  data/h2_summary.csv – 3 rows   (one per N)

Usage
-----
  python experiments/run_h2.py          # from sna_framework/
  cd experiments && python run_h2.py    # from experiments/
"""

from __future__ import annotations

import sys
import os
import time
import warnings

# ---------------------------------------------------------------------------
# Path setup — importable from sna_framework/ or experiments/
# ---------------------------------------------------------------------------
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
for _p in [_FRAMEWORK_ROOT, _EXPERIMENTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from tqdm import tqdm

import networkx as nx   # only for is_connected — centrality via engine

from engine.graph_generator import GraphGenerator
from engine.full_recompute import FullRecompute
from engine.lba import LandmarkApproximation

# ---------------------------------------------------------------------------
# Experimental parameters
# ---------------------------------------------------------------------------
GRAPH_SIZES: list[int] = [50_000, 75_000, 100_000]
N_RUNS: int = 5
BASE_SEED: int = 42
BA_M: int = 2
LANDMARK_FRACTION: float = 0.05
MAX_CONNECT_RETRIES: int = 3       # max seed retries if G is disconnected

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(_FRAMEWORK_ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, "h2_raw.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "h2_summary.csv")

# Raw CSV columns (ordered)
RAW_COLUMNS: list[str] = [
    "N", "run_idx", "seed", "n_landmarks", "landmark_pct",
    "fr_elapsed_ms", "lba_init_ms", "lba_elapsed_ms", "speedup_ratio",
    "pearson_r", "pearson_p", "rmse", "nrmse", "h2_criterion",
]

# Summary CSV columns (ordered)
SUMMARY_COLUMNS: list[str] = [
    "N", "mean_pearson_r", "std_pearson_r", "min_pearson_r",
    "mean_rmse", "mean_nrmse",
    "mean_fr_ms", "mean_lba_ms", "mean_speedup",
    "all_meet_target", "h2_verdict",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a clearly delimited section header (ASCII-safe)."""
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


def classify_h2(min_pearson_r: float) -> str:
    """Classify H2 verdict from the worst-case (minimum) Pearson r.

    Parameters
    ----------
    min_pearson_r : float
        Smallest Pearson r observed across all runs for a given N.

    Returns
    -------
    str
        ``"ACCEPTED"`` if min r > 0.95,
        ``"ACCEPTED_PARTIAL"`` if 0.90 < min r <= 0.95,
        ``"REJECTED"`` if min r <= 0.90.
    """
    if min_pearson_r > 0.95:
        return "ACCEPTED"
    elif min_pearson_r > 0.90:
        return "ACCEPTED_PARTIAL"
    else:
        return "REJECTED"


def _generate_connected_graph(
    gen: GraphGenerator,
    N: int,
    m: int,
    base_seed: int,
    run_idx: int,
) -> tuple[nx.Graph, int]:
    """Generate a BA graph, retrying with seed+100 if disconnected.

    Parameters
    ----------
    gen : GraphGenerator
        Shared generator instance.
    N : int
        Number of nodes.
    m : int
        BA attachment parameter.
    base_seed : int
        Primary seed (42 + run_idx).
    run_idx : int
        Current run index (for logging).

    Returns
    -------
    tuple[nx.Graph, int]
        (G, seed_used) — the connected graph and the seed that produced it.
        Raises RuntimeError if all retries are exhausted.
    """
    seed = base_seed
    for attempt in range(MAX_CONNECT_RETRIES + 1):
        G = gen.generate_ba_graph(N=N, m=m, seed=seed)
        if nx.is_connected(G):
            if attempt > 0:
                print(
                    f"  [WARNING] Graph connected after retry "
                    f"(attempt {attempt}, seed={seed})."
                )
            return G, seed
        print(
            f"  [WARNING] N={N} run={run_idx} seed={seed} "
            f"produced disconnected graph — retrying with seed={seed + 100}."
        )
        seed += 100

    raise RuntimeError(
        f"Could not generate a connected graph for N={N} "
        f"after {MAX_CONNECT_RETRIES} retries."
    )


def _compute_metrics(
    fr_values: np.ndarray,
    lba_values: np.ndarray,
    run_idx: int,
    N: int,
) -> tuple[float, float, float, float]:
    """Compute Pearson r, p, RMSE, and NRMSE between FR and LBA vectors.

    Parameters
    ----------
    fr_values : np.ndarray
        Exact closeness scores (FR), aligned by sorted node order.
    lba_values : np.ndarray
        Approximated closeness scores (LBA), same order as fr_values.
    run_idx : int
        Current run index (for warning messages).
    N : int
        Graph size (for warning messages).

    Returns
    -------
    tuple
        (pearson_r, pearson_p, rmse, nrmse)
    """
    # Pearson r — guard against constant arrays (e.g., degenerate graph)
    try:
        pearson_r, pearson_p = scipy_stats.pearsonr(fr_values, lba_values)
    except ValueError:
        print(
            f"  [WARNING] Constant array detected in run {run_idx} N={N} "
            "— pearson_r set to NaN.",
            file=sys.stderr,
        )
        pearson_r, pearson_p = float("nan"), float("nan")

    # RMSE
    rmse: float = float(np.sqrt(np.mean((fr_values - lba_values) ** 2)))

    # NRMSE — normalize by FR score range
    score_range: float = float(fr_values.max() - fr_values.min())
    if score_range > 0:
        nrmse: float = rmse / score_range
    else:
        print(
            f"  [WARNING] FR score_range == 0 in run {run_idx} N={N} "
            "— nrmse set to NaN.",
            file=sys.stderr,
        )
        nrmse = float("nan")

    return pearson_r, pearson_p, rmse, nrmse


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_h2_experiment() -> None:
    """Run the full H2 experiment: 5 runs x 3 graph sizes.

    For each (N, run_idx) combination:
      1. Generate a connected BA graph.
      2. Run FullRecompute for exact closeness centrality.
      3. Run LandmarkApproximation (time init + compute separately).
      4. Compute Pearson r, RMSE, NRMSE on aligned node vectors.
      5. Collect results to CSV.
    Prints a formatted summary table and per-run Pearson r at the end.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    gen = GraphGenerator()
    fr_engine = FullRecompute()

    all_raw_rows: list[dict] = []

    # -----------------------------------------------------------------------
    # Outer loops: N x run_idx
    # Total iterations = 3 x 5 = 15
    # -----------------------------------------------------------------------
    total_runs = len(GRAPH_SIZES) * N_RUNS
    overall_pbar = tqdm(
        total=total_runs,
        desc="  H2 overall",
        unit="run",
        ncols=72,
        file=sys.stderr,
    )

    for N in GRAPH_SIZES:
        _banner(f"N={N:,}  (5 runs x seed 42..46)")

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            print(f"\n  -- N={N:,}, run={run_idx}, seed={seed} --")

            # ----------------------------------------------------------------
            # Step 1: Generate connected BA graph
            # ----------------------------------------------------------------
            try:
                G, seed_used = _generate_connected_graph(
                    gen, N=N, m=BA_M, base_seed=seed, run_idx=run_idx
                )
            except RuntimeError as exc:
                print(f"  [ERROR] {exc} — skipping run.", file=sys.stderr)
                all_raw_rows.append({
                    "N": N, "run_idx": run_idx, "seed": seed,
                    "n_landmarks": None, "landmark_pct": None,
                    "fr_elapsed_ms": None, "lba_init_ms": None,
                    "lba_elapsed_ms": None, "speedup_ratio": None,
                    "pearson_r": None, "pearson_p": None,
                    "rmse": None, "nrmse": None,
                    "h2_criterion": "SKIPPED",
                })
                overall_pbar.update(1)
                continue

            # ----------------------------------------------------------------
            # Step 2: Full Recompute (exact baseline)
            # ----------------------------------------------------------------
            print(f"  [FR] Computing full closeness (N={N:,}) ...")
            fr_result = fr_engine.compute(G)
            fr_centrality: dict[int, float] = fr_result["centrality"]
            fr_ms: float = fr_result["elapsed_ms"]
            print(f"  [FR] Done: {fr_ms:,.1f} ms")

            # ----------------------------------------------------------------
            # Step 3: LBA — time __init__ (BFS precompute) separately
            # ----------------------------------------------------------------
            print(
                f"  [LBA] Initialising with {LANDMARK_FRACTION*100:.0f}% "
                f"landmarks (N={N:,}) ..."
            )
            t_lba_init_start = time.perf_counter()
            lba = LandmarkApproximation(
                G, landmark_fraction=LANDMARK_FRACTION, seed=seed_used
            )
            lba_init_ms: float = (
                time.perf_counter() - t_lba_init_start
            ) * 1_000.0
            print(f"  [LBA] Init (BFS precompute): {lba_init_ms:,.1f} ms")

            # LBA compute_approximation() — this is what speedup_ratio uses
            print(f"  [LBA] Computing approximation ...")
            lba_result = lba.compute_approximation()
            lba_centrality: dict[int, float] = lba_result["centrality"]
            lba_ms: float = lba_result["elapsed_ms"]
            n_landmarks: int = lba_result["n_landmarks"]
            print(f"  [LBA] Approximation: {lba_ms:,.1f} ms")

            # Warn if LBA (compute pass only) is slower than FR
            if lba_ms > fr_ms:
                print(
                    f"  [WARNING] LBA compute_approximation ({lba_ms:.1f} ms) "
                    f"is SLOWER than FR ({fr_ms:.1f} ms) at N={N}. "
                    "Landmark precomputation may dominate — check lba_init_ms.",
                    file=sys.stderr,
                )

            # ----------------------------------------------------------------
            # Step 4: Accuracy metrics on aligned node vectors
            # ----------------------------------------------------------------
            nodes: list[int] = sorted(G.nodes())
            fr_values = np.array([fr_centrality[n] for n in nodes])
            lba_values = np.array([lba_centrality[n] for n in nodes])

            pearson_r, pearson_p, rmse, nrmse = _compute_metrics(
                fr_values, lba_values, run_idx=run_idx, N=N
            )

            # Speedup: only compute_approximation() elapsed (not lba_init_ms)
            if lba_ms > 0:
                speedup_ratio: float = fr_ms / lba_ms
            else:
                print(
                    f"  [WARNING] lba_elapsed_ms == 0 at N={N} run={run_idx} "
                    "— speedup set to inf.",
                    file=sys.stderr,
                )
                speedup_ratio = float("inf")

            # Classify this run
            if np.isnan(pearson_r):
                h2_criterion = "INVALID_RUN"
            elif pearson_r > 0.95:
                h2_criterion = "MEETS_TARGET"
            else:
                h2_criterion = "BELOW_TARGET"

            r_display = f"{pearson_r:.4f}" if not np.isnan(pearson_r) else "NaN"
            print(
                f"  [RESULT] pearson_r={r_display}  "
                f"rmse={rmse:.6f}  nrmse={nrmse:.4f}  "
                f"speedup={speedup_ratio:.2f}x  "
                f"criterion={h2_criterion}"
            )

            # ----------------------------------------------------------------
            # Step 5: Collect row
            # ----------------------------------------------------------------
            all_raw_rows.append({
                "N": N,
                "run_idx": run_idx,
                "seed": seed_used,
                "n_landmarks": n_landmarks,
                "landmark_pct": round(n_landmarks / N * 100, 4),
                "fr_elapsed_ms": fr_ms,
                "lba_init_ms": lba_init_ms,
                "lba_elapsed_ms": lba_ms,
                "speedup_ratio": speedup_ratio,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "rmse": rmse,
                "nrmse": nrmse,
                "h2_criterion": h2_criterion,
            })

            overall_pbar.update(1)

        # End of 5 runs for this N
        print()

    overall_pbar.close()

    # -----------------------------------------------------------------------
    # Aggregate summaries per N
    # -----------------------------------------------------------------------
    _banner("COMPUTING SUMMARIES")

    all_summary_rows: list[dict] = []
    df_raw = pd.DataFrame(all_raw_rows, columns=RAW_COLUMNS)

    for N in GRAPH_SIZES:
        subset = df_raw[
            (df_raw["N"] == N) &
            (df_raw["h2_criterion"].isin(["MEETS_TARGET", "BELOW_TARGET"]))
        ]

        if subset.empty:
            print(f"  [WARNING] No valid runs for N={N:,} — all skipped/invalid.")
            all_summary_rows.append({
                "N": N,
                "mean_pearson_r": float("nan"),
                "std_pearson_r": float("nan"),
                "min_pearson_r": float("nan"),
                "mean_rmse": float("nan"),
                "mean_nrmse": float("nan"),
                "mean_fr_ms": float("nan"),
                "mean_lba_ms": float("nan"),
                "mean_speedup": float("nan"),
                "all_meet_target": False,
                "h2_verdict": "REJECTED",
            })
            continue

        mean_r = float(subset["pearson_r"].mean())
        std_r  = float(subset["pearson_r"].std(ddof=1)) if len(subset) > 1 else 0.0
        min_r  = float(subset["pearson_r"].min())

        all_meet = bool((subset["h2_criterion"] == "MEETS_TARGET").all())

        summary_row = {
            "N": N,
            "mean_pearson_r": mean_r,
            "std_pearson_r": std_r,
            "min_pearson_r": min_r,
            "mean_rmse": float(subset["rmse"].mean()),
            "mean_nrmse": float(subset["nrmse"].mean()),
            "mean_fr_ms": float(subset["fr_elapsed_ms"].mean()),
            "mean_lba_ms": float(subset["lba_elapsed_ms"].mean()),
            "mean_speedup": float(subset["speedup_ratio"].mean()),
            "all_meet_target": all_meet,
            "h2_verdict": classify_h2(min_r),
        }
        all_summary_rows.append(summary_row)
        print(
            f"  N={N:>7,}  mean_r={mean_r:.4f}  std_r={std_r:.4f}  "
            f"min_r={min_r:.4f}  verdict={summary_row['h2_verdict']}"
        )

    # -----------------------------------------------------------------------
    # Save output files
    # -----------------------------------------------------------------------
    _banner("SAVING OUTPUT FILES")

    df_raw.to_csv(RAW_CSV, index=False)
    print(f"  Saved {len(df_raw)} rows -> {RAW_CSV}")

    df_summary = pd.DataFrame(all_summary_rows, columns=SUMMARY_COLUMNS)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  Saved {len(df_summary)} rows -> {SUMMARY_CSV}")

    # -----------------------------------------------------------------------
    # Final console summary table
    # -----------------------------------------------------------------------
    _banner("H2 EXPERIMENT RESULTS")

    # Column widths
    cw = {"N": 9, "runs": 5, "r": 8, "std": 7, "min": 7, "nrmse": 7, "v": 17}
    hdr = (
        f"  {'N':<{cw['N']}} | {'Runs':>{cw['runs']}} | "
        f"{'Mean r':>{cw['r']}} | {'Std r':>{cw['std']}} | "
        f"{'Min r':>{cw['min']}} | {'NRMSE':>{cw['nrmse']}} | "
        f"{'Verdict':<{cw['v']}}"
    )
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for row in all_summary_rows:
        n_valid = len(
            df_raw[
                (df_raw["N"] == row["N"]) &
                (df_raw["h2_criterion"].isin(["MEETS_TARGET", "BELOW_TARGET"]))
            ]
        )

        def _fmt(v: object, fmt: str = ".4f") -> str:
            """Format float or return 'N/A' for NaN/None."""
            try:
                return format(float(v), fmt)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "N/A"

        print(
            f"  {row['N']:<{cw['N']},} | "
            f"{n_valid:>{cw['runs']}d} | "
            f"{_fmt(row['mean_pearson_r']):>{cw['r']}} | "
            f"{_fmt(row['std_pearson_r']):>{cw['std']}} | "
            f"{_fmt(row['min_pearson_r']):>{cw['min']}} | "
            f"{_fmt(row['mean_nrmse']):>{cw['nrmse']}} | "
            f"{row['h2_verdict']:<{cw['v']}}"
        )

    print(sep)
    print(f"\n  Pearson r target: > 0.95\n")

    # Per-run Pearson r detail
    print("  Per-run Pearson r detail:")
    for N in GRAPH_SIZES:
        runs_subset = df_raw[df_raw["N"] == N]
        r_values = []
        for _, row in runs_subset.iterrows():
            r = row["pearson_r"]
            r_values.append(f"run{int(row['run_idx'])}={r:.4f}" if not pd.isna(r) else f"run{int(row['run_idx'])}=NaN")
        print(f"  N={N:>7,}:  {'  '.join(r_values)}")

    print(
        f"\n  Files saved:\n"
        f"    {RAW_CSV}\n"
        f"    {SUMMARY_CSV}"
    )
    print()

    # -----------------------------------------------------------------------
    # H2 Overall verdict
    # -----------------------------------------------------------------------
    _banner("H2 VERDICT")

    verdicts = [r["h2_verdict"] for r in all_summary_rows]
    if all(v == "ACCEPTED" for v in verdicts):
        verdict = "H2 ACCEPTED -- Pearson r > 0.95 for ALL tested N > 50,000."
    elif all(v in ("ACCEPTED", "ACCEPTED_PARTIAL") for v in verdicts):
        verdict = (
            "H2 ACCEPTED_PARTIAL -- Pearson r > 0.90 for all N,\n"
            "                       but min r <= 0.95 for at least one N.\n"
            "                       Discuss landmark fraction vs accuracy tradeoff."
        )
    else:
        verdict = (
            "H2 REJECTED -- min Pearson r <= 0.90 for at least one N > 50,000.\n"
            "               Consider increasing landmark_fraction."
        )

    print(f"  {verdict}")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_h2_experiment()
