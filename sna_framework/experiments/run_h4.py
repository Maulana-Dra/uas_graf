"""
run_h4.py
=========
H4 Experiment: Closeness-seeded vs Degree-seeded information propagation
in a disaster scenario with random node failures.

Hypothesis H4
-------------
The node with highest Closeness Centrality spreads emergency messages
faster than the node with highest Degree Centrality when random node
failures occur.

Acceptance criteria
-------------------
  ACCEPTED : Closeness-seeded propagation reaches 50% coverage in
             significantly fewer steps than Degree-seeded (p < 0.05,
             one-sided Mann-Whitney U, alternative='less')
  REJECTED : No significant difference, or Degree-seeded is faster.

Propagation model
-----------------
  Simple Independent Cascade (SIC):
    - One seed node activated at step 0
    - Each active node independently activates each inactive neighbour
      with probability p_spread = 0.30 per step
    - Metric: steps until >= 50% of LCC nodes are active
    - Censored at max_steps=500 (value = 501 recorded)

Failure scenario
----------------
  - failure_rate fraction of nodes removed randomly before each run
  - Simulation runs on the Largest Connected Component (LCC)
  - Seed candidates re-identified on LCC after failures

Experimental design
-------------------
  Graph sizes   : 1,000 / 5,000 / 10,000
  Failure rates : 5% / 10% / 20%
  Repetitions   : 50 runs per (N, failure_rate)
  Graph seeds   : {1000: 42, 5000: 43, 10000: 44}
  Failure seeds : 100 + run_idx (reproducible failure sets)
  SIC seeds     : 200 + run_idx (closeness) / 1200 + run_idx (degree)

Output files
------------
  data/h4_raw.csv     – up to 450 rows
  data/h4_summary.csv – 9 rows

Usage
-----
  python experiments/run_h4.py          # from sna_framework/
  cd experiments && python run_h4.py    # from experiments/
"""

from __future__ import annotations

import math
import os
import random
import sys
import warnings

# ---------------------------------------------------------------------------
# Path setup
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

import networkx as nx  # used for: LCC, closeness_centrality (setup only)

from engine.graph_generator import GraphGenerator

# ---------------------------------------------------------------------------
# Experimental parameters
# ---------------------------------------------------------------------------
GRAPH_SIZES: list[int] = [1_000, 5_000, 10_000]
GRAPH_SEEDS: dict[int, int] = {1_000: 42, 5_000: 43, 10_000: 44}
FAILURE_RATES: list[float] = [0.05, 0.10, 0.20]
N_RUNS: int = 50
BA_M: int = 2

# SIC parameters
P_SPREAD: float = 0.30
MAX_STEPS: int = 500
COVERAGE_PCT: float = 0.50
CENSORED_VALUE: int = MAX_STEPS + 1   # = 501

# Seed offsets
FAILURE_SEED_OFFSET: int = 100   # failure_seed = 100 + run_idx
SIC_SEED_CLOSENESS_OFFSET: int = 200    # sic_seed = 200 + run_idx
SIC_SEED_DEGREE_OFFSET: int = 1200     # degree gets 1000 more → independent

# Guard
MIN_LCC_SIZE: int = 10

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(_FRAMEWORK_ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, "h4_raw.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "h4_summary.csv")

RAW_COLUMNS: list[str] = [
    "N", "failure_rate", "run_idx", "lcc_size", "n_removed",
    "closeness_seed", "closeness_score", "degree_seed", "degree_score",
    "same_node", "steps_closeness", "steps_degree",
    "closeness_censored", "degree_censored", "closeness_faster",
]

SUMMARY_COLUMNS: list[str] = [
    "N", "failure_rate", "n_valid_runs",
    "mean_steps_closeness", "mean_steps_degree",
    "median_steps_closeness", "median_steps_degree",
    "pct_closeness_faster", "pct_same_node",
    "closeness_censored_pct", "degree_censored_pct",
    "test_used", "test_statistic", "p_value",
    "significant", "h4_verdict",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a clearly delimited section header (ASCII-safe)."""
    bar = "=" * 66
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


def get_lcc(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as an independent nx.Graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph (may be disconnected).

    Returns
    -------
    nx.Graph
        A fresh copy of the subgraph induced by the LCC nodes.
    """
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def apply_failures(
    G: nx.Graph,
    failure_rate: float,
    seed: int,
) -> tuple[nx.Graph, list]:
    """Randomly remove a fraction of nodes and their incident edges.

    Parameters
    ----------
    G : nx.Graph
        Original graph (NOT modified).
    failure_rate : float
        Fraction of nodes to remove.
    seed : int
        Random seed for reproducible failure sets.

    Returns
    -------
    tuple[nx.Graph, list]
        (G_failed, removed_nodes) where G_failed is a new graph
        with the sampled nodes (and their edges) removed.
    """
    random.seed(seed)
    nodes = list(G.nodes())
    n_remove = int(failure_rate * len(nodes))
    removed = random.sample(nodes, n_remove)

    G_failed = G.copy()
    G_failed.remove_nodes_from(removed)
    return G_failed, removed


def run_sic(
    G_sub: nx.Graph,
    seed_node: int,
    p_spread: float,
    max_steps: int,
    coverage_pct: float,
) -> int:
    """Simulate Simple Independent Cascade from seed_node on G_sub.

    At each step, every node activated in the PREVIOUS step tries to
    activate each of its inactive neighbours independently with
    probability p_spread.  Stops when coverage_pct of all nodes in
    G_sub are active, or propagation dies, or max_steps is reached.

    Parameters
    ----------
    G_sub : nx.Graph
        The graph on which to run the cascade (LCC after failures).
    seed_node : int
        The starting (initially activated) node.
    p_spread : float
        Per-edge activation probability each step.
    max_steps : int
        Hard step limit; returns max_steps + 1 if not reached.
    coverage_pct : float
        Target fraction of G_sub nodes to activate.

    Returns
    -------
    int
        Number of steps to reach target coverage, or max_steps + 1
        if coverage was not reached (censored observation).
    """
    n_remaining: int = G_sub.number_of_nodes()
    target: int = math.ceil(coverage_pct * n_remaining)

    active: set[int] = {seed_node}
    newly_active: set[int] = {seed_node}

    for step in range(1, max_steps + 1):
        next_new: set[int] = set()
        for node in newly_active:
            for neighbour in G_sub.neighbors(node):
                if neighbour not in active:
                    if random.random() < p_spread:
                        next_new.add(neighbour)

        # Only nodes not already active join this wave
        newly_active = next_new - active
        active |= newly_active

        if len(active) >= target:
            return step                  # coverage reached

        if len(newly_active) == 0:
            return max_steps + 1         # propagation died out

    return max_steps + 1                 # hard step limit reached


def classify_h4(
    p_value: float,
    mean_closeness: float,
    mean_degree: float,
) -> str:
    """Classify H4 verdict from test p-value and mean step counts.

    Parameters
    ----------
    p_value : float
        p-value from Mann-Whitney U (one-sided) or T-Test.
    mean_closeness : float
        Mean steps-to-coverage for closeness-seeded runs.
    mean_degree : float
        Mean steps-to-coverage for degree-seeded runs.

    Returns
    -------
    str
        ``"ACCEPTED"`` if p < 0.05 AND closeness is faster (fewer steps).
        ``"REJECTED_NOT_SIGNIFICANT"`` if p >= 0.05.
        ``"REJECTED_DEGREE_FASTER"`` if p < 0.05 but degree is faster.
    """
    if p_value < 0.05 and mean_closeness < mean_degree:
        return "ACCEPTED"
    elif p_value >= 0.05:
        return "REJECTED_NOT_SIGNIFICANT"
    else:
        return "REJECTED_DEGREE_FASTER"


def _statistical_test(
    arr_c: np.ndarray,
    arr_d: np.ndarray,
) -> tuple[str, float, float]:
    """Choose and run the appropriate two-sample test.

    Decision rule:
      - Run Shapiro-Wilk on BOTH arrays.
      - If BOTH pass (p >= 0.05): Independent T-Test.
      - Otherwise: Mann-Whitney U (alternative='less').

    Parameters
    ----------
    arr_c : np.ndarray
        Steps-to-coverage for closeness-seeded runs.
    arr_d : np.ndarray
        Steps-to-coverage for degree-seeded runs.

    Returns
    -------
    tuple[str, float, float]
        (test_name, test_statistic, p_value)
    """
    # Shapiro-Wilk requires at least 3 samples; default to Mann-Whitney if not
    use_ttest = False
    if len(arr_c) >= 3 and len(arr_d) >= 3:
        try:
            _, p_c = scipy_stats.shapiro(arr_c)
            _, p_d = scipy_stats.shapiro(arr_d)
            use_ttest = (p_c >= 0.05) and (p_d >= 0.05)
        except Exception:
            use_ttest = False

    if use_ttest:
        stat, p_val = scipy_stats.ttest_ind(arr_c, arr_d)
        test_name = "Independent T-Test"
    else:
        # Mann-Whitney U, one-sided: alternative='less' tests whether
        # closeness steps are stochastically LESS than degree steps
        # (i.e. closeness reaches coverage in fewer steps = is faster)
        try:
            stat, p_val = scipy_stats.mannwhitneyu(
                arr_c, arr_d, alternative="less"
            )
            test_name = "Mann-Whitney U"
        except ValueError:
            # e.g. all values identical — no discrimination possible
            print(
                "  [WARNING] mannwhitneyu raised ValueError "
                "(all values identical?) — setting p_value=1.0",
                file=sys.stderr,
            )
            stat, p_val = float("nan"), 1.0
            test_name = "Mann-Whitney U"

    return test_name, float(stat), float(p_val)


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_h4_experiment() -> None:
    """Run the full H4 experiment: 50 runs x 3 failure rates x 3 graph sizes.

    For each (N, failure_rate, run_idx):
      1. Apply random node failures to a fixed BA graph.
      2. Extract the Largest Connected Component.
      3. Identify closeness- and degree-based seed nodes on LCC.
      4. Run SIC from each seed (independent random seeds).
      5. Compare steps-to-coverage distributions with statistical tests.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    gen = GraphGenerator()
    all_raw_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    # -----------------------------------------------------------------------
    # Outer loop: graph sizes
    # -----------------------------------------------------------------------
    for N in GRAPH_SIZES:
        graph_seed = GRAPH_SEEDS[N]
        _banner(f"N={N:,}  (graph_seed={graph_seed})")

        G_base = gen.generate_ba_graph(N=N, m=BA_M, seed=graph_seed)
        stats = gen.get_graph_stats(G_base)
        print(
            f"  Graph: nodes={stats['n_nodes']:,}  edges={stats['n_edges']:,}  "
            f"avg_deg={stats['avg_degree']:.3f}  "
            f"connected={stats['is_connected']}"
        )

        # -------------------------------------------------------------------
        # Middle loop: failure rates
        # -------------------------------------------------------------------
        for failure_rate in FAILURE_RATES:
            print(
                f"\n  -- failure_rate={failure_rate*100:.0f}%  "
                f"(50 runs) --"
            )

            closeness_steps_all: list[int] = []
            degree_steps_all: list[int] = []
            combo_raw_rows: list[dict] = []
            skipped_runs: int = 0

            pbar_label = f"  N={N:>6,} fail={failure_rate*100:.0f}%"
            with tqdm(
                total=N_RUNS,
                desc=pbar_label,
                unit="run",
                ncols=72,
                file=sys.stderr,
            ) as pbar:

                for run_idx in range(N_RUNS):
                    failure_seed = FAILURE_SEED_OFFSET + run_idx
                    sic_seed_c   = SIC_SEED_CLOSENESS_OFFSET + run_idx
                    sic_seed_d   = SIC_SEED_DEGREE_OFFSET + run_idx

                    # ---- a. Apply failures ---------------------------------
                    G_failed, removed = apply_failures(
                        G_base, failure_rate, seed=failure_seed
                    )

                    # ---- b. Get LCC ----------------------------------------
                    G_lcc = get_lcc(G_failed)
                    lcc_size = G_lcc.number_of_nodes()

                    if lcc_size < MIN_LCC_SIZE:
                        print(
                            f"\n  [WARNING] LCC size={lcc_size} < "
                            f"{MIN_LCC_SIZE} at run {run_idx} "
                            f"(N={N}, fail={failure_rate}) — skipping.",
                            file=sys.stderr,
                        )
                        skipped_runs += 1
                        pbar.update(1)
                        continue

                    # ---- c. Identify seed candidates on LCC ----------------
                    # nx.closeness_centrality allowed here (simulation setup)
                    closeness_dict: dict[int, float] = (
                        nx.closeness_centrality(G_lcc)
                    )
                    closeness_seed: int = max(
                        closeness_dict, key=closeness_dict.get  # type: ignore[arg-type]
                    )
                    closeness_score: float = closeness_dict[closeness_seed]

                    degree_dict: dict[int, int] = dict(G_lcc.degree())
                    degree_seed: int = max(
                        degree_dict, key=degree_dict.get  # type: ignore[arg-type]
                    )
                    degree_score: int = degree_dict[degree_seed]

                    same_node: bool = closeness_seed == degree_seed

                    # ---- d. SIC from closeness seed ------------------------
                    random.seed(sic_seed_c)
                    steps_closeness: int = run_sic(
                        G_lcc,
                        seed_node=closeness_seed,
                        p_spread=P_SPREAD,
                        max_steps=MAX_STEPS,
                        coverage_pct=COVERAGE_PCT,
                    )

                    # ---- e. SIC from degree seed ---------------------------
                    # Use sic_seed_d = sic_seed_c + 1000 for independence
                    random.seed(sic_seed_d)
                    steps_degree: int = run_sic(
                        G_lcc,
                        seed_node=degree_seed,
                        p_spread=P_SPREAD,
                        max_steps=MAX_STEPS,
                        coverage_pct=COVERAGE_PCT,
                    )

                    # ---- f. Record ----------------------------------------
                    closeness_steps_all.append(steps_closeness)
                    degree_steps_all.append(steps_degree)

                    combo_raw_rows.append({
                        "N": N,
                        "failure_rate": failure_rate,
                        "run_idx": run_idx,
                        "lcc_size": lcc_size,
                        "n_removed": len(removed),
                        "closeness_seed": closeness_seed,
                        "closeness_score": closeness_score,
                        "degree_seed": degree_seed,
                        "degree_score": degree_score,
                        "same_node": same_node,
                        "steps_closeness": steps_closeness,
                        "steps_degree": steps_degree,
                        "closeness_censored": steps_closeness > MAX_STEPS,
                        "degree_censored": steps_degree > MAX_STEPS,
                        "closeness_faster": steps_closeness < steps_degree,
                    })

                    pbar.update(1)

            # End of 50 runs for this (N, failure_rate)
            all_raw_rows.extend(combo_raw_rows)
            n_valid = len(closeness_steps_all)

            print(
                f"\n  Valid runs: {n_valid} / {N_RUNS}  "
                f"(skipped: {skipped_runs})"
            )

            # ----------------------------------------------------------------
            # Statistical test
            # ----------------------------------------------------------------
            if n_valid == 0:
                print(
                    f"  [WARNING] All 50 runs skipped for "
                    f"N={N} failure={failure_rate} — INSUFFICIENT_DATA.",
                    file=sys.stderr,
                )
                all_summary_rows.append({
                    "N": N,
                    "failure_rate": failure_rate,
                    "n_valid_runs": 0,
                    "mean_steps_closeness": float("nan"),
                    "mean_steps_degree": float("nan"),
                    "median_steps_closeness": float("nan"),
                    "median_steps_degree": float("nan"),
                    "pct_closeness_faster": float("nan"),
                    "pct_same_node": float("nan"),
                    "closeness_censored_pct": float("nan"),
                    "degree_censored_pct": float("nan"),
                    "test_used": "N/A",
                    "test_statistic": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "h4_verdict": "INSUFFICIENT_DATA",
                })
                continue

            arr_c = np.array(closeness_steps_all, dtype=float)
            arr_d = np.array(degree_steps_all, dtype=float)

            test_name, test_stat, p_val = _statistical_test(arr_c, arr_d)

            mean_c = float(np.mean(arr_c))
            mean_d = float(np.mean(arr_d))
            median_c = float(np.median(arr_c))
            median_d = float(np.median(arr_d))

            pct_closeness_faster = float(
                np.mean([r["closeness_faster"] for r in combo_raw_rows]) * 100
            )
            pct_same_node = float(
                np.mean([r["same_node"] for r in combo_raw_rows]) * 100
            )
            closeness_censored_pct = float(
                np.mean([r["closeness_censored"] for r in combo_raw_rows]) * 100
            )
            degree_censored_pct = float(
                np.mean([r["degree_censored"] for r in combo_raw_rows]) * 100
            )

            significant = p_val < 0.05
            verdict = classify_h4(p_val, mean_c, mean_d)

            print(
                f"  Test: {test_name}  stat={test_stat:.4f}  "
                f"p={p_val:.4f}  sig={significant}\n"
                f"  mean_steps: closeness={mean_c:.1f}  "
                f"degree={mean_d:.1f}\n"
                f"  C-faster={pct_closeness_faster:.1f}%  "
                f"same_node={pct_same_node:.1f}%  "
                f"verdict={verdict}"
            )

            all_summary_rows.append({
                "N": N,
                "failure_rate": failure_rate,
                "n_valid_runs": n_valid,
                "mean_steps_closeness": mean_c,
                "mean_steps_degree": mean_d,
                "median_steps_closeness": median_c,
                "median_steps_degree": median_d,
                "pct_closeness_faster": pct_closeness_faster,
                "pct_same_node": pct_same_node,
                "closeness_censored_pct": closeness_censored_pct,
                "degree_censored_pct": degree_censored_pct,
                "test_used": test_name,
                "test_statistic": test_stat,
                "p_value": p_val,
                "significant": significant,
                "h4_verdict": verdict,
            })

    # -----------------------------------------------------------------------
    # Save output files
    # -----------------------------------------------------------------------
    _banner("SAVING OUTPUT FILES")

    df_raw = pd.DataFrame(all_raw_rows, columns=RAW_COLUMNS)
    df_raw.to_csv(RAW_CSV, index=False)
    print(f"  Saved {len(df_raw)} rows -> {RAW_CSV}")

    df_summary = pd.DataFrame(all_summary_rows, columns=SUMMARY_COLUMNS)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  Saved {len(df_summary)} rows -> {SUMMARY_CSV}")

    # -----------------------------------------------------------------------
    # Console summary table
    # -----------------------------------------------------------------------
    _banner("H4 EXPERIMENT RESULTS -- DISASTER SCENARIO")

    cw = {
        "N": 7, "fail": 6, "mc": 14, "md": 14,
        "cf": 10, "pv": 8, "verd": 26,
    }
    hdr = (
        f"  {'N':<{cw['N']}} | {'Fail%':<{cw['fail']}} | "
        f"{'Mean C-steps':>{cw['mc']}} | {'Mean D-steps':>{cw['md']}} | "
        f"{'C-faster%':>{cw['cf']}} | {'p-val':>{cw['pv']}} | "
        f"{'Verdict':<{cw['verd']}}"
    )
    sep = "  " + "-" * (len(hdr) - 2)

    print(hdr)
    print(sep)

    for row in all_summary_rows:
        def _f(v: object, fmt: str = ".2f") -> str:
            try:
                return format(float(v), fmt)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return "N/A"

        fail_pct = f"{float(row['failure_rate'])*100:.0f}%"
        sig_mark = "*" if row.get("significant") else " "
        pv_str = f"{_f(row['p_value'], '.4f')}{sig_mark}"

        print(
            f"  {row['N']:<{cw['N']},} | "
            f"{fail_pct:<{cw['fail']}} | "
            f"{_f(row['mean_steps_closeness']):>{cw['mc']}} | "
            f"{_f(row['mean_steps_degree']):>{cw['md']}} | "
            f"{_f(row['pct_closeness_faster'], '.1f'):>{cw['cf']}} | "
            f"{pv_str:>{cw['pv']}} | "
            f"{str(row['h4_verdict']):<{cw['verd']}}"
        )

    print(sep)
    print(
        "\n  (* = statistically significant at alpha=0.05)\n"
        "  NOTE: steps > 500 = censored (propagation failed to reach 50%)\n"
        "  Test: Mann-Whitney U (alternative='less') unless both groups\n"
        "        pass Shapiro-Wilk -> Independent T-Test\n"
    )

    # Same-node overlap detail
    print("  Same-node overlap (closeness_seed == degree_seed):")
    for N in GRAPH_SIZES:
        parts = []
        for fr in FAILURE_RATES:
            match = [
                r for r in all_summary_rows
                if r["N"] == N and r["failure_rate"] == fr
            ]
            if match and not pd.isna(match[0].get("pct_same_node", float("nan"))):
                parts.append(
                    f"fail{int(fr*100)}={match[0]['pct_same_node']:.1f}%"
                )
            else:
                parts.append(f"fail{int(fr*100)}=N/A")
        print(f"  N={N:>6,}:  {'  '.join(parts)}")

    # Censored-data summary
    print("\n  Censored runs (propagation did not reach 50% coverage):")
    print(
        f"  {'N':<8} {'Fail%':<7} "
        f"{'C-censor%':>10} {'D-censor%':>10}"
    )
    for row in all_summary_rows:
        fail_pct = f"{float(row['failure_rate'])*100:.0f}%"
        cc = (
            f"{row['closeness_censored_pct']:.1f}%"
            if not pd.isna(row.get("closeness_censored_pct", float("nan")))
            else "N/A"
        )
        dc = (
            f"{row['degree_censored_pct']:.1f}%"
            if not pd.isna(row.get("degree_censored_pct", float("nan")))
            else "N/A"
        )
        print(
            f"  {row['N']:<8,} {fail_pct:<7} "
            f"{cc:>10} {dc:>10}"
        )

    print(
        f"\n  Files saved:\n"
        f"    {RAW_CSV}\n"
        f"    {SUMMARY_CSV}"
    )
    print()

    # -----------------------------------------------------------------------
    # H4 Overall verdict
    # -----------------------------------------------------------------------
    _banner("H4 VERDICT")

    verdicts = [r["h4_verdict"] for r in all_summary_rows]
    n_accepted = verdicts.count("ACCEPTED")
    n_total = len(verdicts)

    if n_accepted == n_total:
        verdict_line = (
            "H4 ACCEPTED -- Closeness-seeded propagation significantly\n"
            "               faster than Degree-seeded across ALL (N, failure) combos."
        )
    elif n_accepted > n_total // 2:
        verdict_line = (
            f"H4 PARTIAL ACCEPT -- ACCEPTED in {n_accepted}/{n_total} combos.\n"
            "                     Closeness advantage is context-dependent."
        )
    elif n_accepted > 0:
        verdict_line = (
            f"H4 WEAK -- ACCEPTED in only {n_accepted}/{n_total} combos.\n"
            "           Consider pct_same_node as a key confound."
        )
    else:
        verdict_line = (
            "H4 REJECTED -- Closeness-seeded NOT significantly faster\n"
            "               than Degree-seeded in any tested scenario."
        )

    print(f"  {verdict_line}")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_h4_experiment()
