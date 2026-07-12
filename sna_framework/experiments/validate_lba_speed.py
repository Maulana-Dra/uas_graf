"""
validate_lba_speed.py
=====================
Standalone validation script to check LandmarkApproximation (LBA) performance
before committing to the full H2 experiment run.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
import sys
import tracemalloc

# Path setup — make engine importable regardless of working directory
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _EXPERIMENTS_DIR.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from engine.graph_generator import GraphGenerator
from engine.full_recompute import FullRecompute
from engine.lba import LandmarkApproximation


def validate_lba_speed() -> None:
    print("=" * 70)
    print("LBA SPEED & MEMORY VALIDATION — PRE-H2 CHECK")
    print("=" * 70)
    print("Testing LandmarkApproximation at increasing N before ")
    print("committing to the full H2 run (N=50k/75k/100k x 5 runs).")
    print("Expected total runtime for this validation: ~20-25 minutes ")
    print("(includes running FullRecompute at N=50,000 and N=100,000 ")
    print("for correctness comparison, in addition to LBA itself).")
    print()

    TEST_SCALES = [10_000, 50_000, 100_000]
    TIME_BUDGET_PER_SCALE_SECONDS = 600  # 10 minutes max per scale before flagging as "too slow"

    results = []
    
    for N in TEST_SCALES:
        print(f"\n{'='*70}")
        print(f"Testing N={N:,}")
        print(f"{'='*70}")
        
        print(f"[1/3] Generating BA graph (N={N:,}, m=2, seed=42)...")
        t0 = time.perf_counter()
        try:
            G = GraphGenerator().generate_ba_graph(N=N, m=2, seed=42)
            gen_time = time.perf_counter() - t0
            print(f"  Done in {gen_time:.1f}s")
        except Exception as e:
            print(f"  [ERROR] Error generating graph: {e}")
            break
        
        n_landmarks = max(10, int(0.05 * N))
        print(f"  Landmark count (5% of N, min 10): {n_landmarks:,}")
        
        print("[2/3] Initializing LandmarkApproximation (includes ")
        print("      landmark BFS precomputation)...")
        
        tracemalloc.start()
        t0 = time.perf_counter()
        
        try:
            lba = LandmarkApproximation(G, landmark_fraction=0.05, seed=42)
            init_time = time.perf_counter() - t0
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"  Init done in {init_time:.1f}s ({init_time/60:.1f} min)")
            print(f"  Peak memory during init: {peak_mem / 1024 / 1024:.1f} MB")
            
            if init_time > TIME_BUDGET_PER_SCALE_SECONDS:
                print(f"  [WARNING] Init exceeded time budget "
                      f"({TIME_BUDGET_PER_SCALE_SECONDS}s). This scale "
                      f"is likely infeasible without optimization.")
        
        except MemoryError:
            tracemalloc.stop()
            print(f"  [ERROR] MemoryError during LBA init at N={N:,}!")
            print("  This confirms LBA needs the same chunking treatment ")
            print("  applied to ICC before H2 can run at this scale.")
            results.append({
                "N": N, "status": "MEMORY_ERROR", "init_time_s": None,
                "compute_time_s": None, "peak_mem_mb": None
            })
            print("\n  Skipping remaining scales — MemoryError indicates ")
            print("  the same fix pattern as ICC is needed first.")
            break
        
        except Exception as e:
            tracemalloc.stop()
            print(f"  [ERROR] Unexpected error during LBA init: {e}")
            results.append({
                "N": N, "status": "ERROR", "error": str(e),
                "init_time_s": None, "compute_time_s": None
            })
            break
        
        print("[3/3] Running compute_approximation()...")
        t0 = time.perf_counter()
        try:
            lba_result = lba.compute_approximation()
            compute_time = time.perf_counter() - t0
            print(f"  Compute done in {compute_time:.1f}s")
            print(f"  Top5: {lba_result['top5']}")
        except MemoryError:
            print(f"  [ERROR] MemoryError during compute_approximation() at N={N:,}!")
            results.append({
                "N": N, "status": "MEMORY_ERROR_COMPUTE", 
                "init_time_s": init_time, "compute_time_s": None,
                "peak_mem_mb": peak_mem / 1024 / 1024
            })
            break
        except Exception as e:
            print(f"  [ERROR] Unexpected error during compute_approximation(): {e}")
            results.append({
                "N": N, "status": "ERROR_COMPUTE", "error": str(e),
                "init_time_s": init_time, "compute_time_s": None,
                "peak_mem_mb": peak_mem / 1024 / 1024
            })
            break
        
        # Correctness check against FR
        print(f"\n  Running FR for correctness check (N={N:,})...")
        fr_t0 = time.perf_counter()
        fr_result = FullRecompute().compute(G)
        fr_time = time.perf_counter() - fr_t0
        print(f"  FR completed in {fr_time:.1f}s")
        
        import random
        import numpy as np
        
        random.seed(42)
        # Sample more nodes at larger N for a more representative check, 
        # but cap it to keep the comparison itself fast
        sample_size = min(50, N)
        sample_nodes = random.sample(list(G.nodes()), sample_size)
        
        diffs = []
        large_diff_nodes = []
        for node in sample_nodes:
            fr_val = fr_result['centrality'].get(node, 0.0)
            lba_val = lba_result['centrality'].get(node, 0.0)
            diff = abs(fr_val - lba_val)
            diffs.append(diff)
            if diff > 0.05:  # same threshold used in ICC validation
                large_diff_nodes.append((node, fr_val, lba_val, diff))
        
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        
        # Also compute Pearson r across the FULL node set (not just the 
        # sample), since this is what H2 actually reports — this is the 
        # metric that matters most for validating H2 readiness
        from scipy.stats import pearsonr
        nodes_sorted = sorted(G.nodes())
        fr_values = np.array([fr_result['centrality'][n] for n in nodes_sorted])
        lba_values = np.array([lba_result['centrality'][n] for n in nodes_sorted])
        
        try:
            pearson_r, pearson_p = pearsonr(fr_values, lba_values)
        except Exception as e:
            pearson_r, pearson_p = float('nan'), float('nan')
            print(f"   Could not compute Pearson r: {e}")
        
        correctness_note = (
            f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"pearson_r={pearson_r:.4f}, large_diff_count={len(large_diff_nodes)}/{sample_size}, "
            f"FR_time={fr_time:.1f}s"
        )
        print(f"  Correctness: {correctness_note}")
        
        if pearson_r < 0.90:
            print(f"   [CRITICAL] Pearson r={pearson_r:.4f} is below 0.90 — ")
            print("     this suggests a BUG in the vectorized rewrite, ")
            print("     NOT normal approximation variance. Do NOT proceed ")
            print("     to the full H2 run until this is investigated.")
        elif pearson_r < 0.95:
            print(f"   [WARNING] Pearson r={pearson_r:.4f} is below the ")
            print("     H2 target threshold (0.95). This may be normal ")
            print("     approximation behavior at this N, or may indicate ")
            print("     a regression — compare against previous validation ")
            print("     results if available before proceeding.")
        else:
            print(f"   Pearson r={pearson_r:.4f} meets H2 target threshold.")
        
        results.append({
            "N": N,
            "status": "OK",
            "n_landmarks": n_landmarks,
            "gen_time_s": gen_time,
            "init_time_s": init_time,
            "compute_time_s": compute_time,
            "peak_mem_mb": peak_mem / 1024 / 1024,
            "fr_time_s": fr_time,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "large_diff_count": len(large_diff_nodes),
            "sample_size": sample_size
        })
        
        # Early exit if this scale already took very long — no point 
        # attempting the next (larger) scale
        total_this_scale = init_time + compute_time
        if total_this_scale > TIME_BUDGET_PER_SCALE_SECONDS:
            print(f"\n  [WARNING] N={N:,} took {total_this_scale/60:.1f} min total ")
            print(f"  (exceeds {TIME_BUDGET_PER_SCALE_SECONDS/60:.0f} min budget). ")
            print("  Skipping larger scales — optimization needed first.")
            break

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'N':<12} {'Status':<12} {'Init(s)':<10} {'Compute(s)':<10} "
          f"{'PeakMB':<10} {'FR(s)':<10} {'Pearson_r':<12}")
    print("-" * 90)
    for r in results:
        n = r.get("N", "?")
        status = r.get("status", "?")
        init_s = r.get("init_time_s")
        comp_s = r.get("compute_time_s")
        mem = r.get("peak_mem_mb")
        fr_s = r.get("fr_time_s")
        pearson_r = r.get("pearson_r")
        
        init_str = f"{init_s:.1f}" if init_s is not None else "N/A"
        comp_str = f"{comp_s:.1f}" if comp_s is not None else "N/A"
        mem_str = f"{mem:.1f}" if mem is not None else "N/A"
        fr_str = f"{fr_s:.1f}" if fr_s is not None else "N/A"
        pearson_str = f"{pearson_r:.4f}" if pearson_r is not None else "N/A"
        
        print(f"{n:<12} {status:<12} {init_str:<10} {comp_str:<10} "
              f"{mem_str:<10} {fr_str:<10} {pearson_str:<12}")

    successful = [r for r in results if r.get("status") == "OK"]
    print("\n" + "=" * 70)
    print("H2 FULL RUN PROJECTION")
    print("=" * 70)
    
    if len(successful) == len(TEST_SCALES):
        total_projected_s = 0
        for r in successful:
            per_run_s = r["init_time_s"] + r["compute_time_s"]
            # H2 spec: 5 runs per N, but only N=50k/75k/100k are in 
            # scope (10k was just a sanity baseline, not part of H2)
            if r["N"] in [50_000, 100_000]:
                runs_for_this_n = 5
                n_projected_s = per_run_s * runs_for_this_n
                total_projected_s += n_projected_s
                print(f"N={r['N']:,}: ~{per_run_s:.1f}s/run × 5 runs = "
                      f"~{n_projected_s/3600:.2f} hours")
        print("Note: N=75,000 not directly tested (interpolate between ")
        print("50k and 100k results, or add 75k to TEST_SCALES if precise ")
        print("estimate needed).")
        print(f"\nProjected total (N=50k + N=100k only): "
              f"~{total_projected_s/3600:.2f} hours")
        
        all_pearson_ok = all(
            r.get("pearson_r", 0) > 0.90 for r in successful
        )
        
        if all_pearson_ok:
            print("\n RECOMMENDATION: All scales passed BOTH performance ")
            print("   AND correctness checks (Pearson r > 0.90 at all N). ")
            print("   Safe to proceed with the full H2 run.")
        else:
            print("\n RECOMMENDATION: Performance is fine, but correctness ")
            print("   FAILED at one or more scales (Pearson r <= 0.90). ")
            print("   Do NOT run the full H2 experiment yet — investigate ")
            print("   the vectorized compute_approximation() rewrite for bugs.")
    else:
        print("\n [WARNING] Could not complete all test scales — see errors above.")
        print("RECOMMENDATION: LBA requires optimization (igraph backend ")
        print("+ chunked distance computation, same pattern as ICC fix) ")
        print("before attempting the full H2 run. Do NOT run run_h2.py ")
        print("at full scale yet.")

    data_dir = _FRAMEWORK_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    log_path = data_dir / "lba_validation_log.json"
    with open(log_path, "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)
    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    validate_lba_speed()
