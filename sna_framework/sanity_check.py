"""
sanity_check.py
===============
End-to-end validation script for the SNA Framework engine.

Runs all three closeness-centrality methods on a small BA graph (N=500),
applies 3 batch updates via ICC, and prints a final comparison table.

Usage
-----
    python sanity_check.py          # from inside sna_framework/
    python sna_framework/sanity_check.py   # from project root
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Make sure the engine package is importable when running from either
# the sna_framework/ directory or its parent.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from scipy.stats import pearsonr  # type: ignore[import-untyped]

from engine.graph_generator import GraphGenerator
from engine.full_recompute import FullRecompute
from engine.icc import IncrementalCloseness
from engine.lba import LandmarkApproximation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_NODES: int = 500
M_EDGES: int = 2
SEED: int = 42
CHURN_RATE: float = 0.05
N_BATCHES: int = 3
LANDMARK_FRACTION: float = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    """Print a clearly delimited section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def _print_top5(label: str, top5: list[tuple[int, float]]) -> None:
    """Pretty-print top-5 nodes with their centrality scores."""
    print(f"  [{label}] Top-5 nodes (node_id, score):")
    for rank, (node, score) in enumerate(top5, start=1):
        print(f"    #{rank}  node {node:>4d}  ->  {score:.6f}")


# ---------------------------------------------------------------------------
# Main sanity-check routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete sanity-check suite and print a summary table."""

    gen = GraphGenerator()
    fr_engine = FullRecompute()

    # -----------------------------------------------------------------------
    # Step 1 — Generate BA graph
    # -----------------------------------------------------------------------
    _banner("STEP 1 · Generate BA Graph  (N=500, m=2, seed=42)")
    G = gen.generate_ba_graph(N=N_NODES, m=M_EDGES, seed=SEED)

    stats = gen.get_graph_stats(G)
    print("  Graph statistics:")
    for key, val in stats.items():
        print(f"    {key:<15s}: {val}")
    assert stats["is_connected"], "ERROR: Initial graph is NOT connected!"

    # -----------------------------------------------------------------------
    # Step 2 — Full Recompute (baseline, before batches)
    # -----------------------------------------------------------------------
    _banner("STEP 2 · FullRecompute (initial)")
    fr_initial = fr_engine.compute(G)
    _print_top5("FR-initial", fr_initial["top5"])
    print(f"  Elapsed: {fr_initial['elapsed_ms']:.3f} ms")

    # -----------------------------------------------------------------------
    # Step 3 — Initialise ICC
    # -----------------------------------------------------------------------
    _banner("STEP 3 · Initialise IncrementalCloseness")
    icc = IncrementalCloseness(G)
    # ICC holds its own internal copy of G; we will keep our G in sync
    # by applying batches to both (G is also needed for FR re-check later).

    # -----------------------------------------------------------------------
    # Steps 4 & 5 — Apply 3 batches via ICC
    # -----------------------------------------------------------------------
    icc_result_last = None
    for batch_num in range(1, N_BATCHES + 1):
        _banner(f"STEP 4-5 · Batch {batch_num} of {N_BATCHES}")

        # Plan the batch (do NOT apply yet)
        batch = gen.generate_batch_updates(
            G,
            churn_rate=CHURN_RATE,
            seed=SEED + batch_num,  # different seed each round
        )

        # Apply to our reference G so FR can re-check after batch 3
        gen.apply_batch(G, batch)

        # Verify connectivity after every apply
        if not gen.get_graph_stats(G)["is_connected"]:
            print(f"  WARNING: Graph disconnected after batch {batch_num}!")
        else:
            print(f"  [OK] Graph remains connected after batch {batch_num}.")

        # Pass the SAME batch plan to ICC (ICC applies it internally)
        icc_result_last = icc.update(batch)
        _print_top5(f"ICC-batch{batch_num}", icc_result_last["top5"])
        print(
            f"  n_affected: {icc_result_last['n_affected']} | "
            f"elapsed: {icc_result_last['elapsed_ms']:.3f} ms"
        )

    # -----------------------------------------------------------------------
    # Step 6 — FullRecompute after batch 3  (compare with ICC)
    # -----------------------------------------------------------------------
    _banner("STEP 6 · FullRecompute after Batch 3  (comparison)")
    fr_final = fr_engine.compute(G)
    _print_top5("FR-final", fr_final["top5"])
    print(f"  Elapsed: {fr_final['elapsed_ms']:.3f} ms")

    print("\n  [Comparison] ICC top5 vs FR-final top5:")
    print(f"  {'Rank':<5} {'ICC node':>10} {'ICC score':>12} "
          f"{'FR node':>10} {'FR score':>12}")
    for i, ((icc_n, icc_s), (fr_n, fr_s)) in enumerate(
        zip(icc_result_last["top5"], fr_final["top5"]), start=1
    ):
        match = "[=]" if icc_n == fr_n else "[!]"
        print(
            f"  #{i:<4} {icc_n:>10d} {icc_s:>12.6f} "
            f"{fr_n:>10d} {fr_s:>12.6f}  {match}"
        )

    # -----------------------------------------------------------------------
    # Step 7 — Initialise LBA
    # -----------------------------------------------------------------------
    _banner("STEP 7 · Initialise LandmarkApproximation (landmark_fraction=0.05)")
    lba = LandmarkApproximation(G, landmark_fraction=LANDMARK_FRACTION, seed=SEED)
    print(f"  Landmark IDs (first 10): {lba.get_landmark_ids()[:10]} …")

    # -----------------------------------------------------------------------
    # Step 8 — LBA approximation
    # -----------------------------------------------------------------------
    _banner("STEP 8 · LBA compute_approximation()")
    lba_result = lba.compute_approximation()
    _print_top5("LBA", lba_result["top5"])
    print(
        f"  Landmarks used: {lba_result['n_landmarks']} | "
        f"elapsed: {lba_result['elapsed_ms']:.3f} ms"
    )

    # -----------------------------------------------------------------------
    # Step 9 — Pearson r (FR vs LBA)
    # -----------------------------------------------------------------------
    _banner("STEP 9 · Pearson Correlation  FR-final vs LBA")
    common_nodes = sorted(
        set(fr_final["centrality"].keys()) & set(lba_result["centrality"].keys())
    )
    fr_vals = [fr_final["centrality"][n] for n in common_nodes]
    lba_vals = [lba_result["centrality"][n] for n in common_nodes]

    pearson_r, p_value = pearsonr(fr_vals, lba_vals)
    print(f"  Pearson r (FR vs LBA): {pearson_r:.4f}  (p={p_value:.2e})")

    if pearson_r >= 0.95:
        print("  [OK] r >= 0.95  -- within H2 target range.")
    elif pearson_r >= 0.85:
        print(
            "  [!]  0.85 <= r < 0.95  -- acceptable for N=500; "
            "H2 target tested at N>50 000."
        )
    else:
        print("  [!!] r < 0.85  -- consider increasing landmark_fraction.")

    # -----------------------------------------------------------------------
    # Step 10 — Final summary table
    # -----------------------------------------------------------------------
    _banner("STEP 10 · Final Summary Table")

    fr_top1_node, fr_top1_score = fr_final["top5"][0]
    icc_top1_node, icc_top1_score = icc_result_last["top5"][0]
    lba_top1_node, lba_top1_score = lba_result["top5"][0]

    col_w = [15, 12, 12, 14]
    header = (
        f"  {'Method':<{col_w[0]}} "
        f"{'Top1 Node':>{col_w[1]}} "
        f"{'Top1 Score':>{col_w[2]}} "
        f"{'Elapsed (ms)':>{col_w[3]}}"
    )
    sep = "  " + "-" * (sum(col_w) + 3 * len(col_w))
    print(header)
    print(sep)

    rows = [
        ("FullRecompute", fr_top1_node, fr_top1_score, fr_final["elapsed_ms"]),
        ("ICC (batch3)", icc_top1_node, icc_top1_score, icc_result_last["elapsed_ms"]),
        ("LBA", lba_top1_node, lba_top1_score, lba_result["elapsed_ms"]),
    ]
    for method, node, score, ms in rows:
        print(
            f"  {method:<{col_w[0]}} "
            f"{node:>{col_w[1]}d} "
            f"{score:>{col_w[2]}.6f} "
            f"{ms:>{col_w[3]}.3f}"
        )

    print()
    print("  [OK] sanity_check.py completed successfully.")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
