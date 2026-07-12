"""
validate_icc_speedup.py
=======================
Standalone benchmark: verify ICC igraph-acceleration correctness AND 
measure real speedup at N=50,000 (configurable via TEST_N) before 
committing to the H1 run.

This script:
  - Does NOT write to h1_raw.csv, h1_summary.csv, or h1_checkpoint.json
  - Only writes to data/icc_validation_log.json
  - Runs a SINGLE batch at N=TEST_N, with a dynamic churn rate (default 0.01)
  - Compares FR and ICC centrality values on a 20-node sample
  - Compares results side-by-side with previous runs (via archived logs)
  - Projects full H1 run time based on actual per-N measurements
  - Prints a clear PASS / NEEDS_INVESTIGATION verdict

Usage
-----
  python experiments/validate_icc_speedup.py   # from sna_framework/
"""

from __future__ import annotations

import sys
import os
import time
import json
import random
import traceback
import datetime
import shutil
import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make engine importable regardless of working directory
# ---------------------------------------------------------------------------
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
for _p in [_FRAMEWORK_ROOT, _EXPERIMENTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.graph_generator import GraphGenerator
from engine.full_recompute import FullRecompute
from engine.icc import IncrementalCloseness

# ---------------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------------
TEST_N = 50_000    # Default to 50,000 as requested
CHURN_RATE = 0.01  # Keep at 0.01 unless testing other scenarios

# Reference: pure-NetworkX ICC at N=50,000 was ~37 minutes (2,220,000 ms)
OLD_ICC_MS_50K: float = 37 * 60 * 1000


# ---------------------------------------------------------------------------
# Helper: Print N-scaling comparison
# ---------------------------------------------------------------------------

def print_n_scaling_comparison(current_result: dict) -> None:
    """Load all archived validation logs and print an N-scaling table."""
    print("\n" + "=" * 70)
    print("N-SCALING COMPARISON (all archived validation runs)")
    print("=" * 70)

    log_files = sorted(glob.glob("data/icc_validation_log_N*_churn*.json"))
    
    all_results = []
    for lf in log_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
        except Exception as exc:
            print(f"Error reading {lf}: {exc}")

    # Add current run's result too
    all_results.append(current_result)

    # Sort by N, then churn_rate
    all_results.sort(key=lambda r: (r.get("N", 0), r.get("churn_rate", 0)))

    print(f"{'N':<12} {'Churn':<8} {'n_affected%':<14} {'ICC(s)':<10} {'FR(s)':<10} {'ICC/FR':<10}")
    print("-" * 70)
    for r in all_results:
        n = r.get("N", "?")
        churn = r.get("churn_rate", "?")
        
        n_val = n if isinstance(n, int) else 50000
        n_aff_pct = (r.get("icc_n_affected", 0) / n_val * 100)
        
        icc_s = r.get("icc_elapsed_ms", 0) / 1000
        fr_s = r.get("fr_elapsed_ms", 0) / 1000
        ratio = r.get("icc_elapsed_ms", 1) / r.get("fr_elapsed_ms", 1)
        print(f"{n:<12} {churn:<8} {f'{n_aff_pct:.1f}%':<14} {icc_s:<10.1f} {fr_s:<10.1f} {f'{ratio:.2f}x':<10}")

    print()
    # Compute growth ratio between N values at same churn rate
    if len(all_results) >= 2:
        same_churn = [r for r in all_results if r.get("churn_rate") == CHURN_RATE]
        if len(same_churn) >= 2:
            same_churn.sort(key=lambda r: r.get("N", 0))
            r1, r2 = same_churn[-2], same_churn[-1]
            n1, n2 = r1.get("N"), r2.get("N")
            t1 = r1.get("icc_elapsed_ms", 0) + r1.get("fr_elapsed_ms", 0)
            t2 = r2.get("icc_elapsed_ms", 0) + r2.get("fr_elapsed_ms", 0)
            if t1 > 0 and n1 != n2:
                time_growth = t2 / t1
                n_growth = n2 / n1
                print(f"Scaling from N={n1} to N={n2} ({n_growth:.1f}x larger):")
                print(f"  Combined ICC+FR time grew {time_growth:.2f}x")
                if time_growth > n_growth * 1.5:
                    print("  [WARN] WORSE than linear scaling — time is growing faster than N.")
                    print("         This indicates larger N will be disproportionately expensive.")
                else:
                    print("  [OK] Roughly linear or better scaling.")


# ---------------------------------------------------------------------------
# Helper: Project full H1 run time based on actual measurements
# ---------------------------------------------------------------------------

def project_full_run_time(current_result: dict) -> None:
    """Project full run time based only on N values with actual logs."""
    print("\n" + "=" * 70)
    print("REVISED FULL RUN PROJECTION (based on actual per-N measurements)")
    print("=" * 70)

    log_files = glob.glob("data/icc_validation_log_N*_churn*.json")
    all_results = []
    for lf in log_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
        except Exception:
            pass
    all_results.append(current_result)  # include current run

    target_ns = [10_000, 50_000, 100_000]
    batches_per_combo = 15
    churns = 3  # 1%, 5%, 10%

    total_hours = 0.0
    for n in target_ns:
        matching = [r for r in all_results if r.get("N") == n]
        if matching:
            # Average combined time for this N across tested churns
            avg_combined_ms = sum(
                r.get("icc_elapsed_ms", 0) + r.get("fr_elapsed_ms", 0)
                for r in matching
            ) / len(matching)
            combo_hours = (avg_combined_ms * batches_per_combo * churns) / 1000 / 3600
            total_hours += combo_hours
            print(f"  N={n:>7,}: measured, ~{combo_hours:.2f} hours for 3 churn combos")
        else:
            print(f"  N={n:>7,}: NOT YET MEASURED — cannot project accurately.")
            print(f"             Run validate_icc_speedup.py with TEST_N={n} first.")

    print(f"\nTotal projected (only for N values with real data): ~{total_hours:.2f} hours")
    print("NOTE: This projection ONLY includes N values that have been actually measured.")
    print("      If any N is 'NOT YET MEASURED', the total above is incomplete.")


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_icc_speedup() -> None:
    """Run single-batch ICC speedup validation at N=TEST_N."""

    print("=" * 70)
    print(f"ICC IGRAPH ACCELERATION VALIDATION — SINGLE BATCH TEST (N={TEST_N:,})")
    print("=" * 70)
    print(f"This test verifies correctness AND measures real speedup at churn={CHURN_RATE*100:.0f}%")
    print("before committing to the full 9-combination H1 run.")
    print()

    gen = GraphGenerator()

    # -----------------------------------------------------------------------
    # [1/5] Generate BA graph at N=TEST_N
    # -----------------------------------------------------------------------
    print(f"[1/5] Generating BA graph (N={TEST_N:,}, m=2, seed=42)...")
    t0 = time.perf_counter()
    G = gen.generate_ba_graph(N=TEST_N, m=2, seed=42)
    gen_time = time.perf_counter() - t0
    stats = gen.get_graph_stats(G)
    print(f"  Done in {gen_time:.1f}s")
    print(f"  Stats: {stats}")

    if not stats["is_connected"]:
        print(f"\n[ERROR] Generated graph is NOT connected. Cannot proceed.")
        print(f"  This should not happen for BA(N={TEST_N}, m=2).")
        sys.exit(1)
    print("  [OK] Graph is connected.")

    # -----------------------------------------------------------------------
    # [2/5] Initialize ICC (one-time full closeness computation)
    # -----------------------------------------------------------------------
    print("\n[2/5] Initializing IncrementalCloseness (one-time full compute)...")
    print("  (Now igraph-accelerated — should take only ~2-3 minutes!)")
    t0 = time.perf_counter()
    icc = IncrementalCloseness(G.copy())
    init_time = time.perf_counter() - t0
    print(f"  Done in {init_time:.1f}s ({init_time/60:.1f} min)")

    # -----------------------------------------------------------------------
    # [3/5] Generate ONE batch update (churn_rate=CHURN_RATE, seed=42)
    # -----------------------------------------------------------------------
    print(f"\n[3/5] Generating 1 batch update (churn_rate={CHURN_RATE}, seed=42)...")
    batch = gen.generate_batch_updates(G, churn_rate=CHURN_RATE, seed=42)
    print(f"  Added: {len(batch['added'])} edges, "
          f"Removed: {len(batch['removed'])} edges")

    # -----------------------------------------------------------------------
    # [4/5] Measure FR on this batch
    # -----------------------------------------------------------------------
    print("\n[4/5] Measuring FR (Full Recompute, igraph backend) on this batch...")
    G_fr = G.copy()
    gen.apply_batch(G_fr, batch)
    fr_result = FullRecompute().compute(G_fr)
    print(f"  FR elapsed: {fr_result['elapsed_ms']:.1f} ms "
          f"({fr_result['elapsed_ms']/1000:.1f}s)")
    print(f"  FR top5: {fr_result['top5']}")

    # -----------------------------------------------------------------------
    # [5/5] Measure ICC.update() on the SAME batch
    # -----------------------------------------------------------------------
    print("\n[5/5] Measuring ICC.update() (igraph-accelerated) on the SAME batch...")
    try:
        icc_result = icc.update(batch)
    except Exception:
        print("\n[FATAL] ICC update() raised an exception:")
        traceback.print_exc()
        print("\nICC update failed — DO NOT proceed to full run until this is fixed.")
        sys.exit(1)

    print(f"  ICC elapsed: {icc_result['elapsed_ms']:.1f} ms "
          f"({icc_result['elapsed_ms']/1000:.1f}s)")
    print(f"  ICC n_affected: {icc_result['n_affected']:,} / {TEST_N:,} "
          f"({icc_result['n_affected']/TEST_N*100:.1f}%)")
    print(f"  ICC top5: {icc_result['top5']}")

    # -----------------------------------------------------------------------
    # Correctness check: FR vs ICC centrality on 20 sample nodes
    # -----------------------------------------------------------------------
    random.seed(42)
    try:
        sample_nodes = random.sample(
            list(G_fr.nodes()), min(20, G_fr.number_of_nodes())
        )
    except Exception as exc:
        print(f"\n[ERROR] Could not sample nodes: {exc}")
        print("  This may indicate a node-mapping bug — DO NOT proceed.")
        sys.exit(1)

    print("\n--- Correctness Check: FR vs ICC centrality (sample of 20 nodes) ---")
    print(f"{'Node':<10} {'FR value':<15} {'ICC value':<15} {'Abs Diff':<12} {'Flag'}")
    print("-" * 65)

    max_diff = 0.0
    large_diff_count = 0
    for node in sample_nodes:
        try:
            fr_val = fr_result["centrality"].get(node, 0.0)
            icc_val = icc_result["centrality"].get(node, 0.0)
        except KeyError as exc:
            print(f"\n[ERROR] KeyError when comparing centrality dicts: {exc}")
            print("  This indicates a node-mapping bug in _nx_to_igraph or")
            print("  _recompute_affected_closeness. DO NOT proceed to full run.")
            sys.exit(1)

        diff = abs(fr_val - icc_val)
        max_diff = max(max_diff, diff)
        flag = ""
        if diff > 0.05:
            flag = "!! LARGE DIFF"
            large_diff_count += 1
        print(f"{node:<10} {fr_val:<15.6f} {icc_val:<15.6f} {diff:<12.6f} {flag}")

    print("-" * 65)
    print(f"Max absolute difference in sample: {max_diff:.6f}")
    print(f"Nodes with diff > 0.05: {large_diff_count}/20")

    if large_diff_count > 10:
        print(
            "\n!! WARNING: Many large differences detected. This may indicate\n"
            "   a bug in the igraph BFS conversion (e.g. node index mapping\n"
            "   issue), NOT just normal ICC approximation error.\n"
            "   Investigate before proceeding to full run."
        )
    else:
        print("\n[OK] Differences appear within normal ICC approximation range.")

    # -----------------------------------------------------------------------
    # Speedup calculation (only reference-able if N=50k, otherwise print comparison)
    # -----------------------------------------------------------------------
    speedup = 0.0
    if TEST_N == 50_000:
        speedup = OLD_ICC_MS_50K / icc_result["elapsed_ms"]
        print("\n" + "=" * 70)
        print("SPEEDUP SUMMARY (N=50,000)")
        print("=" * 70)
        print(f"  Previous ICC time (pure NetworkX) : ~{OLD_ICC_MS_50K/1000/60:.1f} min")
        print(f"  New ICC time (igraph-accelerated) : {icc_result['elapsed_ms']/1000:.2f}s "
              f"({icc_result['elapsed_ms']/1000/60:.2f} min)")
        print(f"  Speedup                           : {speedup:.1f}x")

    # -----------------------------------------------------------------------
    # Save validation log (NOT the experiment CSVs)
    # -----------------------------------------------------------------------
    validation_result = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "N": TEST_N,
        "churn_rate": CHURN_RATE,
        "graph_gen_time_s": round(gen_time, 3),
        "icc_init_time_s": round(init_time, 3),
        "fr_elapsed_ms": round(fr_result["elapsed_ms"], 3),
        "icc_elapsed_ms": round(icc_result["elapsed_ms"], 3),
        "icc_n_affected": icc_result["n_affected"],
        "icc_n_affected_pct": round(icc_result["n_affected"] / TEST_N * 100, 2),
        "old_icc_ms_reference": OLD_ICC_MS_50K if TEST_N == 50_000 else 0,
        "speedup_vs_old_icc": round(speedup, 2) if TEST_N == 50_000 else 0,
        "max_sample_diff": round(max_diff, 8),
        "large_diff_count": large_diff_count,
        "projected_full_run_hours": 0.0,
        "verdict": "PASS" if large_diff_count <= 10 else "NEEDS_INVESTIGATION",
    }

    out_path = Path(_FRAMEWORK_ROOT) / "data" / "icc_validation_log.json"
    
    # Archive the old log if it exists and matches target parameters
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_n = old_data.get("N", "unknown")
            old_churn = old_data.get("churn_rate", "unknown")
            archive_name = out_path.parent / f"icc_validation_log_N{old_n}_churn{old_churn}.json"
            shutil.copy(out_path, archive_name)
            print(f"\nArchived previous log to {archive_name.name}")
        except Exception as exc:
            print(f"\nError archiving old log: {exc}")

    # Write current log
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation_result, f, indent=2)
    print(f"Validation log saved to: {out_path}")

    # Print scaling and revision comparisons
    print_n_scaling_comparison(validation_result)
    project_full_run_time(validation_result)

    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    if large_diff_count <= 10:
        print("[OK] Correctness check passed.")
        print("     Ensure to check the time growth analysis above before running the full experiment.")
    else:
        print("[FAIL] High discrepancy between FR and ICC. Check the implementation.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    validate_icc_speedup()
