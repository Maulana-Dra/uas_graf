#  DO NOT RUN THIS SCRIPT WHILE H2/H4 EXPERIMENTS ARE STILL RUNNING.
# This script is CPU-intensive and will compete for the same limited 
# cores (Intel i3-8130U, 2 core/4 thread), slowing down or 
# potentially destabilizing concurrent experiments.
# 
# Wait until experiments/run_h2_h4_sequence.py has fully completed 
# (check data/h2_h4_sequence_log.txt for "SEQUENCE COMPLETE") before 
# running this script.

"""
run_h1.py
=========
H1 Experiment: ICC vs Full Recompute speed comparison on dynamic BA graphs.

This script aligns with the official exam specification (soal EAS) requiring
exactly 7 combinations and 30 batches per combination. For previously
completed 15-batch combinations, the script automatically supplements them
with 15 more batches (reconstructing graph/ICC state without re-timing),
rather than re-running from scratch.

Features:
  - Controlled auto-stop: stops cleanly after a configured combination finishes.
"""

from __future__ import annotations

import sys
import os
import json
import warnings
import datetime
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make engine importable regardless of working directory
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

import networkx as nx  # only used for is_connected check on G, not centrality

from engine.graph_generator import GraphGenerator
from engine.full_recompute import FullRecompute
from engine.icc import IncrementalCloseness

# ---------------------------------------------------------------------------
# Experimental parameters
# ---------------------------------------------------------------------------
GRAPH_SIZES: list[int] = [10_000, 50_000, 100_000]
CHURN_RATES: list[float] = [0.01, 0.05, 0.10]
N_BATCHES_DEFAULT = 30  # Standard batch count (Wilcoxon/Paired T-Test minimum recommended n>=30)

OFFICIAL_COMBINATIONS = [
    (10_000, 0.01),
    (10_000, 0.05),
    (10_000, 0.10),
    (50_000, 0.01),
    (50_000, 0.05),
    (50_000, 0.10),
    (100_000, 0.05),  # ONLY this combo at N=100,000 per official spec
]

# ============================================================
# TO RESUME REMAINING COMBOS TOMORROW: change the line below to
#   AUTO_STOP_AFTER_COMBO = None
# then simply re-run: python experiments/run_h1.py
# It will skip all completed combos (including N=50,000 churn=0.05
# which will be auto-stopped-at today) and continue with:
#   N=50,000 churn=0.10, then N=100,000 churn=0.05
# ============================================================
AUTO_STOP_AFTER_COMBO = None


def get_n_batches(N: int, churn_rate: float) -> int:
    """All combinations now use the standard batch count per 
    official exam specification.
    """
    return N_BATCHES_DEFAULT


N_WARMUP: int = 1          # warm-up batches to discard before measuring
BASE_SEED: int = 42
BA_M: int = 2
MIN_VALID_ROWS: int = 10   # minimum valid batches to produce a summary row
CHECKPOINT_BACKEND: str = "igraph"      # FR backend identifier for checkpoint
CHECKPOINT_ICC_BACKEND: str = "igraph"  # ICC backend identifier for checkpoint

# ---------------------------------------------------------------------------
# Output paths (relative to framework root, created if missing)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(_FRAMEWORK_ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, "h1_raw.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "h1_summary.csv")
CHECKPOINT_JSON = os.path.join(DATA_DIR, "h1_checkpoint.json")
LIVE_LOG_PATH = os.path.join(DATA_DIR, "h1_live_log.txt")


# ---------------------------------------------------------------------------
# Live logging helper
# ---------------------------------------------------------------------------

def log_live(message: str) -> None:
    """Append a timestamped line to h1_live_log.txt AND print to console.

    Never raises — logging failures must never crash the experiment.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        with open(LIVE_LOG_PATH, "a", encoding="utf-8") as _lf:
            _lf.write(line + "\n")
    except Exception as _log_exc:
        print(f"  (warning: failed to write live log: {_log_exc})")


# ---------------------------------------------------------------------------
# Helper: H1 criterion classifier
# ---------------------------------------------------------------------------

def classify_h1(efficiency_pct: float, churn_rate: float) -> str:
    """Classify a single (N, churn) result against the H1 criterion."""
    if efficiency_pct >= 80.0:
        return "MEETS_TARGET"
    elif efficiency_pct >= 0.0:
        return "BELOW_TARGET"
    else:
        return "ICC_SLOWER"


# ---------------------------------------------------------------------------
# Helper: print a section banner (ASCII-safe for Windows CP1252)
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a clearly delimited section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


# ---------------------------------------------------------------------------
# Helper: suppress noisy engine print statements inside the tqdm loop
# ---------------------------------------------------------------------------

class _SuppressPrint:
    """Context manager that redirects stdout to os.devnull during measurements."""

    def __init__(self) -> None:
        self._original_stdout = sys.stdout

    def __enter__(self) -> "_SuppressPrint":
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, *args: object) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ---------------------------------------------------------------------------
# Helper: connectivity-guaranteed batch generator (local to run_h1.py)
# ---------------------------------------------------------------------------

import random as _random

def _safe_batch(
    G: nx.Graph,
    churn_rate: float,
    add_fraction: float = 0.80,
    seed: int | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Generate a batch of edge changes guaranteed to keep G connected."""
    rng = _random.Random(seed)

    n_changes: int = max(1, int(G.number_of_edges() * churn_rate))
    n_add: int = max(0, int(n_changes * add_fraction))
    n_remove: int = max(0, n_changes - n_add)

    nodes: list[int] = list(G.nodes())

    # ---- Edge additions (same logic as engine) ----------------------------
    added: list[tuple[int, int]] = []
    existing_edges: set[frozenset] = {frozenset(e) for e in G.edges()}
    attempts = 0
    max_attempts = n_add * 20
    while len(added) < n_add and attempts < max_attempts:
        u, v = rng.sample(nodes, 2)
        fs = frozenset({u, v})
        if fs not in existing_edges:
            added.append((u, v))
            existing_edges.add(fs)
        attempts += 1

    # ---- Edge removals — cumulative connectivity testing ------------------
    removed: list[tuple[int, int]] = []
    G_work = G.copy()
    candidate_edges = list(G_work.edges())
    rng.shuffle(candidate_edges)

    for edge in candidate_edges:
        if len(removed) >= n_remove:
            break
        u, v = edge
        if G_work.degree(u) <= 1 or G_work.degree(v) <= 1:
            continue
        G_work.remove_edge(u, v)
        if nx.is_connected(G_work):
            removed.append((u, v))
        else:
            G_work.add_edge(u, v)

    del G_work
    return {"added": added, "removed": removed}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict | None:
    """Load checkpoint from disk. Returns None if file does not exist."""
    if not os.path.exists(CHECKPOINT_JSON):
        return None
    with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(combo_batch_counts: dict[str, int]) -> None:
    """Write/update checkpoint file with current combo batch counts."""
    checkpoint = {
        "combo_batch_counts": combo_batch_counts,
        "backend": CHECKPOINT_BACKEND,
        "icc_backend": CHECKPOINT_ICC_BACKEND,
        "n_batches_per_combo": N_BATCHES_DEFAULT,
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def _append_raw_rows(raw_rows: list[dict], raw_columns: list[str]) -> None:
    """Append raw batch rows to h1_raw.csv (write header only if new file)."""
    if not raw_rows:
        return
    df = pd.DataFrame(raw_rows, columns=raw_columns)
    write_header = not os.path.exists(RAW_CSV)
    df.to_csv(RAW_CSV, mode="a", index=False, header=write_header)


def _append_summary_row(summary_row: dict, summary_columns: list[str]) -> None:
    """Append or overwrite a summary row in h1_summary.csv to keep it unique."""
    df_new = pd.DataFrame([summary_row], columns=summary_columns)
    if os.path.exists(SUMMARY_CSV):
        try:
            df_old = pd.read_csv(SUMMARY_CSV)
            # Drop the row for the same N and churn_rate if it exists to avoid duplicates
            df_old = df_old[~((df_old["N"] == summary_row["N"]) & (df_old["churn_rate"] == summary_row["churn_rate"]))]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_final = df_new
    else:
        df_final = df_new
    df_final.to_csv(SUMMARY_CSV, index=False)


def get_existing_batch_count(N: int, churn_rate: float) -> int:
    """Check how many batches already exist in h1_raw.csv for this combination."""
    if not os.path.exists(RAW_CSV):
        return 0
    try:
        df = pd.read_csv(RAW_CSV)
        matching = df[(df["N"] == N) & (df["churn_rate"] == churn_rate)]
        return len(matching)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# --status command
# ---------------------------------------------------------------------------

def print_status() -> None:
    """Print checkpoint contents and pending combos, then exit."""
    checkpoint = _load_checkpoint()
    all_combos = OFFICIAL_COMBINATIONS
    total = len(all_combos)

    print("\n=== H1 Experiment Status ===")
    if checkpoint is None:
        print(f"  No checkpoint found at: {CHECKPOINT_JSON}")
        print(f"  0/{total} combos completed.")
    else:
        combo_batch_counts = checkpoint.get("combo_batch_counts", {})
        if not combo_batch_counts:
            # Reconstruct from completed_combos list if using old checkpoint format
            completed_list = checkpoint.get("completed_combos", [])
            for combo in completed_list:
                key = f"{combo[0]}_{combo[1]}"
                combo_batch_counts[key] = get_existing_batch_count(combo[0], combo[1])

        print(f"  Checkpoint  : {CHECKPOINT_JSON}")
        print(f"  FR Backend  : {checkpoint.get('backend', 'unknown')}")
        print(f"  ICC Backend : {checkpoint.get('icc_backend', 'unknown')}")
        print(f"  Updated     : {checkpoint.get('last_updated', '?')}")
        
        print("\n  Combinations status:")
        for N, churn in all_combos:
            key = f"{N}_{churn}"
            count = combo_batch_counts.get(key, 0)
            status_str = "COMPLETED" if count >= N_BATCHES_DEFAULT else f"IN PROGRESS ({count}/{N_BATCHES_DEFAULT} batches)"
            print(f"    N={N:>7,}  churn={churn*100:>2.0f}%  ->  {status_str}")
    print()


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_h1_experiment() -> None:
    """Run the H1 experiment across all 7 official (N, churn) combinations."""
    os.makedirs(DATA_DIR, exist_ok=True)
    session_start = datetime.datetime.now()

    # Print auto-stop notice at start
    if AUTO_STOP_AFTER_COMBO is not None:
        log_live(
            f"NOTE: Auto-stop is configured to trigger after "
            f"N={AUTO_STOP_AFTER_COMBO[0]} churn={AUTO_STOP_AFTER_COMBO[1]} completes. "
            f"Remaining combos after that point will NOT run automatically in this session."
        )

    # -----------------------------------------------------------------------
    # Checkpoint / resume guard
    # -----------------------------------------------------------------------
    raw_csv_exists = os.path.exists(RAW_CSV)
    summary_csv_exists = os.path.exists(SUMMARY_CSV)
    checkpoint = _load_checkpoint()

    if checkpoint is None and (raw_csv_exists or summary_csv_exists):
        print(
            "\n[WARNING] h1_raw.csv or h1_summary.csv already exist but "
            "data/h1_checkpoint.json does NOT.\n"
            "  Please manually delete these files before running:\n"
            f"    {RAW_CSV}\n"
            f"    {SUMMARY_CSV}\n"
        )
        sys.exit(1)

    combo_batch_counts: dict[str, int] = {}
    if checkpoint is not None:
        ckpt_backend = checkpoint.get("backend")
        ckpt_icc_backend = checkpoint.get("icc_backend")

        if ckpt_backend != CHECKPOINT_BACKEND or ckpt_icc_backend != CHECKPOINT_ICC_BACKEND:
            print(
                "\n[ERROR] Checkpoint was created with a different backend.\n"
                f"  Checkpoint FR backend  : {ckpt_backend!r}  (expected: {CHECKPOINT_BACKEND!r})\n"
                f"  Checkpoint ICC backend : {ckpt_icc_backend!r}  (expected: {CHECKPOINT_ICC_BACKEND!r})\n"
                "  Delete data/h1_raw.csv, data/h1_summary.csv, and data/h1_checkpoint.json to start fresh.\n"
            )
            sys.exit(1)

        # Migrate old format to combo_batch_counts
        if "combo_batch_counts" in checkpoint:
            combo_batch_counts = checkpoint["combo_batch_counts"]
        else:
            completed_list = checkpoint.get("completed_combos", [])
            for combo in completed_list:
                N, churn = combo[0], combo[1]
                key = f"{N}_{churn}"
                combo_batch_counts[key] = get_existing_batch_count(N, churn)
            _save_checkpoint(combo_batch_counts)
            log_live("Migrated checkpoint schema to combo_batch_counts.")

        # Re-verify and clean counts up against the actual CSV file
        for key in list(combo_batch_counts.keys()):
            try:
                parts = key.split("_")
                N, churn = int(parts[0]), float(parts[1])
                actual_count = get_existing_batch_count(N, churn)
                if combo_batch_counts[key] != actual_count:
                    combo_batch_counts[key] = actual_count
            except Exception:
                pass
        _save_checkpoint(combo_batch_counts)

    log_live(f"Batch scheme: ALL combinations use {N_BATCHES_DEFAULT} batches (no overrides active). Confirmed per updated instructor requirement.")

    # Migrate existing CSV files if they exist but lack the new columns
    if raw_csv_exists:
        try:
            df = pd.read_csv(RAW_CSV)
            any_migrated = False
            if "n_batches_used" not in df.columns:
                df["n_batches_used"] = N_BATCHES_DEFAULT
                any_migrated = True
            if "is_reduced_batch_count" not in df.columns:
                df["is_reduced_batch_count"] = False
                any_migrated = True
            if any_migrated:
                df.to_csv(RAW_CSV, index=False)
                log_live("Migrated existing h1_raw.csv to include new columns.")
        except Exception as e:
            log_live(f"Warning: could not migrate h1_raw.csv: {e}")

    if summary_csv_exists:
        try:
            df = pd.read_csv(SUMMARY_CSV)
            any_migrated = False
            if "n_batches_used" not in df.columns:
                df["n_batches_used"] = N_BATCHES_DEFAULT
                any_migrated = True
            if "is_reduced_batch_count" not in df.columns:
                df["is_reduced_batch_count"] = False
                any_migrated = True
            if "notes" not in df.columns:
                df["notes"] = ""
                any_migrated = True
            if any_migrated:
                df.to_csv(SUMMARY_CSV, index=False)
                log_live("Migrated existing h1_summary.csv to include new columns.")
        except Exception as e:
            log_live(f"Warning: could not migrate h1_summary.csv: {e}")

    # Column definitions
    raw_columns = [
        "N", "churn_rate", "batch_index",
        "fr_elapsed_ms", "icc_elapsed_ms",
        "speedup_ratio", "efficiency_pct",
        "n_affected", "n_nodes", "n_edges",
        "n_batches_used", "is_reduced_batch_count",
    ]
    summary_columns = [
        "N", "churn_rate",
        "mean_fr_ms", "mean_icc_ms",
        "mean_speedup", "mean_efficiency_pct", "std_efficiency_pct",
        "test_used", "test_statistic", "p_value",
        "significant", "h1_criterion",
        "n_batches_used", "is_reduced_batch_count", "notes",
    ]

    gen = GraphGenerator()
    fr_engine = FullRecompute()

    all_summary_rows: list[dict] = []

    # Log overall run start
    all_combos = OFFICIAL_COMBINATIONS
    log_live("H1 EXPERIMENT STARTED (or resumed)")
    log_live(f"Official combinations: {OFFICIAL_COMBINATIONS}")

    # -----------------------------------------------------------------------
    # Loop over the 7 combinations
    # -----------------------------------------------------------------------
    for N, churn_rate in OFFICIAL_COMBINATIONS:
        combo_key = f"{N}_{churn_rate}"
        existing_count = get_existing_batch_count(N, churn_rate)

        # Skip if already completed to target size
        if combo_batch_counts.get(combo_key, 0) >= N_BATCHES_DEFAULT or existing_count >= N_BATCHES_DEFAULT:
            if combo_batch_counts.get(combo_key, 0) < existing_count:
                combo_batch_counts[combo_key] = existing_count
                _save_checkpoint(combo_batch_counts)
            print(
                f"\n  [SKIP] N={N:,} churn={churn_rate*100:.0f}% "
                f"— already completed ({existing_count} batches)."
            )
            
            # Auto-stop check for skipped combo
            if AUTO_STOP_AFTER_COMBO is not None and (N, churn_rate) == AUTO_STOP_AFTER_COMBO:
                log_live(
                    f"=== AUTO-STOP TRIGGERED: N={N} churn={churn_rate} is already "
                    f"completed, which is the configured stop point (AUTO_STOP_AFTER_COMBO). ==="
                )
                remaining_list = [c for c in OFFICIAL_COMBINATIONS if f"{c[0]}_{c[1]}" not in combo_batch_counts]
                log_live(f"Remaining combinations NOT YET started: {remaining_list}")
                log_live("=== EXPERIMENT PAUSED (clean auto-stop, not a crash) ===")
                print("\n" + "=" * 70)
                print("AUTO-STOP: Configured stopping point was already finished.")
                print("Script is exiting cleanly. Data is fully saved.")
                print("=" * 70)
                sys.exit(0)
                
            continue

        batches_needed = N_BATCHES_DEFAULT - existing_count
        log_live(f"N={N} churn={churn_rate}: has {existing_count} batches, need {batches_needed} more to reach {N_BATCHES_DEFAULT}.")

        _banner(f"N={N:,}  |  churn_rate={churn_rate*100:.0f}%")
        log_live(f"=== STARTING N={N} churn={churn_rate} | standard ({N_BATCHES_DEFAULT} batches) ===")

        try:
            # ------------------------------------------------------------
            # Step 1: Generate fresh BA graph
            # ------------------------------------------------------------
            print(f"  Generating BA graph N={N}, m={BA_M}, seed={BASE_SEED} ...")
            G: nx.Graph = gen.generate_ba_graph(N=N, m=BA_M, seed=BASE_SEED)

            stats = gen.get_graph_stats(G)
            print(
                f"  Stats: n_nodes={stats['n_nodes']:,}, "
                f"n_edges={stats['n_edges']:,}, "
                f"avg_degree={stats['avg_degree']:.4f}, "
                f"connected={stats['is_connected']}, "
                f"density={stats['density']:.6f}"
            )

            if not stats["is_connected"]:
                print(f"  WARNING: initial graph N={N} is not connected — skipping combo.")
                continue

            # ------------------------------------------------------------
            # Step 2: Initialise ICC
            # ------------------------------------------------------------
            print(f"  Initialising ICC ...")
            icc = IncrementalCloseness(G.copy())

            # ------------------------------------------------------------
            # Step 3a: Warm-up batch
            # ------------------------------------------------------------
            print(f"  Running {N_WARMUP} warm-up batch(es) [not measured] ...")
            for wu in range(N_WARMUP):
                warmup_seed = BASE_SEED - 1 - wu
                wu_batch = _safe_batch(
                    G, churn_rate=churn_rate, seed=warmup_seed
                )
                with _SuppressPrint():
                    gen.apply_batch(G, wu_batch)

                G_wu = G.copy()
                with _SuppressPrint():
                    gen.apply_batch(G_wu, wu_batch)
                fr_engine.compute(G_wu)
                del G_wu

            # Re-initialise ICC on the post-warmup graph
            print(f"  Re-initialising ICC on post-warmup graph ...")
            icc = IncrementalCloseness(G.copy())

            # ------------------------------------------------------------
            # Step 3b: State Replay (if supplementing)
            # ------------------------------------------------------------
            if existing_count > 0:
                log_live(f"Replaying {existing_count} previously-measured batches to reconstruct graph/ICC state (not re-timed)...")
                for j in range(existing_count):
                    replay_seed = BASE_SEED + j
                    replay_batch = _safe_batch(
                        G, churn_rate=churn_rate, seed=replay_seed
                    )
                    icc.update(replay_batch)
                    with _SuppressPrint():
                        gen.apply_batch(G, replay_batch)

            # Load existing timing rows from CSV
            raw_rows: list[dict] = []
            if existing_count > 0:
                try:
                    df_raw = pd.read_csv(RAW_CSV)
                    matching_raw = df_raw[(df_raw["N"] == N) & (df_raw["churn_rate"] == churn_rate)]
                    raw_rows = matching_raw.to_dict("records")
                    print(f"  Loaded {len(raw_rows)} existing raw rows from {RAW_CSV}")
                except Exception as e:
                    log_live(f"Error loading existing raw rows: {e}")

            raw_rows_new: list[dict] = []
            skipped: int = 0
            batch_times_this_combo: list[float] = []

            pbar_label = f"  N={N:>7,} churn={churn_rate*100:.0f}%"
            with tqdm(
                total=N_BATCHES_DEFAULT,
                initial=existing_count,
                desc=pbar_label,
                unit="batch",
                ncols=80,
                file=sys.stderr,
            ) as pbar:

                for i in range(existing_count, N_BATCHES_DEFAULT):
                    batch_seed = BASE_SEED + i

                    # ---- a. Generate safe batch plan ------------------
                    batch = _safe_batch(
                        G, churn_rate=churn_rate, seed=batch_seed
                    )

                    # ---- b. Measure FR ---------------------------------
                    G_fr = G.copy()
                    with _SuppressPrint():
                        gen.apply_batch(G_fr, batch)
                    fr_result = fr_engine.compute(G_fr)
                    fr_ms: float = fr_result["elapsed_ms"]
                    del G_fr

                    # ---- c. Measure ICC --------------------------------
                    icc_result = icc.update(batch)
                    icc_ms: float = icc_result["elapsed_ms"]
                    n_affected: int = icc_result["n_affected"]

                    # ---- d. Advance main G -----------------------------
                    with _SuppressPrint():
                        gen.apply_batch(G, batch)

                    # ---- Guard: zero ICC timing ------------------------
                    if icc_ms == 0.0:
                        warnings.warn(
                            f"icc_elapsed_ms==0 at batch {i} (N={N}, churn={churn_rate})",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        speedup_ratio = float("inf")
                        efficiency_pct = 100.0
                    else:
                        speedup_ratio = fr_ms / icc_ms
                        efficiency_pct = (1.0 - icc_ms / fr_ms) * 100.0

                    # ---- e. Collect rows --------------------------------
                    row_data = {
                        "N": N,
                        "churn_rate": churn_rate,
                        "batch_index": i,
                        "fr_elapsed_ms": fr_ms,
                        "icc_elapsed_ms": icc_ms,
                        "speedup_ratio": speedup_ratio,
                        "efficiency_pct": efficiency_pct,
                        "n_affected": n_affected,
                        "n_nodes": G.number_of_nodes(),
                        "n_edges": G.number_of_edges(),
                        "n_batches_used": N_BATCHES_DEFAULT,
                        "is_reduced_batch_count": False,
                    }
                    raw_rows.append(row_data)
                    raw_rows_new.append(row_data)

                    # ---- f. Live logging -------------------------------
                    batch_total_s = (fr_ms + icc_ms) / 1000
                    batch_times_this_combo.append(batch_total_s)

                    avg_batch_s = sum(batch_times_this_combo) / len(batch_times_this_combo)
                    batches_remaining = N_BATCHES_DEFAULT - (existing_count + len(batch_times_this_combo))
                    est_remaining_s = avg_batch_s * batches_remaining
                    n_affected_pct = (n_affected / N * 100) if N > 0 else 0

                    log_live(
                        f"N={N} churn={churn_rate} "
                        f"batch={existing_count + len(batch_times_this_combo)}/{N_BATCHES_DEFAULT} | "
                        f"FR={fr_ms/1000:.1f}s | "
                        f"ICC={icc_ms/1000:.1f}s "
                        f"(n_affected={n_affected:,}/{N:,}={n_affected_pct:.1f}%) | "
                        f"batch_total={batch_total_s:.1f}s | "
                        f"est. sisa kombinasi ini: "
                        f"~{est_remaining_s/3600:.2f} jam"
                    )

                    # ---- g. Anomaly warning ----------------------------
                    if len(batch_times_this_combo) >= 3:
                        prev_batches = batch_times_this_combo[:-1]
                        prev_avg = sum(prev_batches) / len(prev_batches)
                        if prev_avg > 0 and batch_total_s > prev_avg * 1.5:
                            log_live(
                                f"  [WARN] Batch ini {batch_total_s:.1f}s, "
                                f"{(batch_total_s/prev_avg - 1)*100:.0f}% "
                                f"lebih lama dari rata-rata batch sebelumnya "
                                f"({prev_avg:.1f}s)."
                            )

                    pbar.update(1)

            n_valid = len(raw_rows)
            print(f"\n  Collected {n_valid} valid rows ({skipped} skipped).")

            # ------------------------------------------------------------
            # Step 4: Statistical test for this combo
            # ------------------------------------------------------------
            fr_times = np.array([r["fr_elapsed_ms"] for r in raw_rows])
            icc_times = np.array([r["icc_elapsed_ms"] for r in raw_rows])
            speedup_ratios = np.array([r["speedup_ratio"] for r in raw_rows])
            efficiency_pcts = np.array([r["efficiency_pct"] for r in raw_rows])

            p_shapiro = float("nan")
            test_name = "N/A"
            test_stat = float("nan")
            p_value = float("nan")
            significant = False
            
            try:
                differences = fr_times - icc_times
                shapiro_stat, p_shapiro = scipy_stats.shapiro(differences)

                if p_shapiro >= 0.05:
                    test_name = "Paired T-Test"
                    test_stat, p_value = scipy_stats.ttest_rel(fr_times, icc_times)
                else:
                    test_name = "Wilcoxon Signed-Rank"
                    test_stat, p_value = scipy_stats.wilcoxon(
                        fr_times, icc_times, alternative="greater"
                    )
                significant = bool(p_value < 0.05)
            except Exception as stats_exc:
                log_live(f"  [WARN] Uji statistik gagal ({stats_exc}). Fallback digunakan.")
                test_name = "INSUFFICIENT_SAMPLE_SIZE"
                test_stat = float("nan")
                p_value = float("nan")
                significant = False

            mean_eff = float(np.mean(efficiency_pcts))
            shapiro_str = f"{p_shapiro:.4f}" if not np.isnan(p_shapiro) else "N/A"
            test_stat_str = f"{test_stat:.4f}" if not np.isnan(test_stat) else "N/A"
            p_val_str = f"{p_value:.6f}" if not np.isnan(p_value) else "N/A"
            print(
                f"  Shapiro-Wilk p={shapiro_str} -> using {test_name}\n"
                f"  test_stat={test_stat_str}, p_value={p_val_str}, significant={significant}\n"
                f"  mean_efficiency_pct={mean_eff:.2f}%  mean_speedup={np.mean(speedup_ratios):.3f}x"
            )

            summary_row = {
                "N": N,
                "churn_rate": churn_rate,
                "mean_fr_ms": float(np.mean(fr_times)),
                "mean_icc_ms": float(np.mean(icc_times)),
                "mean_speedup": float(np.mean(speedup_ratios)),
                "mean_efficiency_pct": mean_eff,
                "std_efficiency_pct": float(np.std(efficiency_pcts)),
                "test_used": test_name,
                "test_statistic": float(test_stat) if not np.isnan(test_stat) else float("nan"),
                "p_value": float(p_value) if not np.isnan(p_value) else float("nan"),
                "significant": significant,
                "h1_criterion": classify_h1(mean_eff, churn_rate),
                "n_batches_used": N_BATCHES_DEFAULT,
                "is_reduced_batch_count": False,
                "notes": "",
            }

            all_summary_rows.append(summary_row)

            combo_total_s = sum(batch_times_this_combo)
            combo_avg_s = combo_total_s / len(batch_times_this_combo) if batch_times_this_combo else 0.0
            log_live(
                f"=== FINISHED N={N} churn={churn_rate} | "
                f"session_added_batches={len(batch_times_this_combo)} | "
                f"avg_batch={combo_avg_s:.1f}s | "
                f"mean_efficiency_pct={mean_eff:.1f}% ==="
            )

            # ------------------------------------------------------------
            # Save checkpoint and rows
            # ------------------------------------------------------------
            _append_raw_rows(raw_rows_new, raw_columns)
            _append_summary_row(summary_row, summary_columns)
            combo_batch_counts[combo_key] = N_BATCHES_DEFAULT
            _save_checkpoint(combo_batch_counts)
            print(
                f"  [SAVED] N={N:,} churn={churn_rate*100:.0f}% -> "
                f"appended to CSVs; checkpoint updated "
                f"({len(combo_batch_counts)}/7 combos done)."
            )

            # Controlled auto-stop check
            if AUTO_STOP_AFTER_COMBO is not None and (N, churn_rate) == AUTO_STOP_AFTER_COMBO:
                log_live(
                    f"=== AUTO-STOP TRIGGERED: Just finished N={N} churn={churn_rate}, "
                    f"which is the configured stop point (AUTO_STOP_AFTER_COMBO). ==="
                )
                remaining_list = [c for c in OFFICIAL_COMBINATIONS if f"{c[0]}_{c[1]}" not in combo_batch_counts]
                log_live(f"Remaining combinations NOT YET started: {remaining_list}")
                log_live("To resume, either:")
                log_live("  (a) Set AUTO_STOP_AFTER_COMBO = None in this script and re-run to process all remaining combos, or")
                log_live("  (b) Change AUTO_STOP_AFTER_COMBO to a later combo if you want another controlled stopping point, or")
                log_live("  (c) Just re-run as-is if you've already set AUTO_STOP_AFTER_COMBO = None — it will resume from the checkpoint automatically.")
                log_live("=== EXPERIMENT PAUSED (clean auto-stop, not a crash) ===")
                print("\n" + "=" * 70)
                print("AUTO-STOP: Finished the configured stopping point combo.")
                print("Script is exiting cleanly. Data is fully saved.")
                print("To continue with remaining combinations, set")
                print("AUTO_STOP_AFTER_COMBO = None and re-run this script.")
                print("=" * 70)
                sys.exit(0)

        except KeyboardInterrupt:
            print(
                f"\n\n  [INTERRUPTED] During N={N:,} churn={churn_rate*100:.0f}%.\n"
                f"  This combo will be redone on next run (partial data NOT saved).\n"
                f"  Progress so far: {len(combo_batch_counts)}/7 combos saved.\n"
            )
            sys.exit(0)

        except Exception as exc:
            print(
                f"\n  [ERROR] Exception during N={N:,} churn={churn_rate*100:.0f}%:\n  {exc}\n"
                f"  Partial data NOT saved. Progress: {len(combo_batch_counts)}/7 combos saved.\n"
            )
            raise

    # -----------------------------------------------------------------------
    # Session completion log
    # -----------------------------------------------------------------------
    session_elapsed = datetime.datetime.now() - session_start
    session_h = session_elapsed.total_seconds() / 3600
    log_live("=== ALL H1 COMBINATIONS COMPLETE ===")
    log_live(f"Total wall time this session: {session_elapsed} (~{session_h:.2f} jam)")

    # -----------------------------------------------------------------------
    # Final console summary table
    # -----------------------------------------------------------------------
    _banner("H1 EXPERIMENT RESULTS")

    if os.path.exists(SUMMARY_CSV):
        df_summary_all = pd.read_csv(SUMMARY_CSV)
        display_rows = df_summary_all.to_dict("records")
    else:
        display_rows = all_summary_rows

    cw = {"N": 8, "churn": 6, "fr": 14, "icc": 14, "eff": 8, "p": 9, "crit": 18}
    hdr = (
        f"  {'N':<{cw['N']}} | {'Churn':<{cw['churn']}} | "
        f"{'Mean FR (ms)':<{cw['fr']}} | {'Mean ICC (ms)':<{cw['icc']}} | "
        f"{'Eff%':<{cw['eff']}} | {'p-value':<{cw['p']}} | "
        f"{'Criterion':<{cw['crit']}}"
    )
    sep = "  " + "-" * (len(hdr) - 2)

    print(hdr)
    print(sep)

    for row in display_rows:
        churn_pct = f"{row['churn_rate']*100:.0f}%"
        mean_fr = f"{row['mean_fr_ms']:.2f}" if not pd.isna(row["mean_fr_ms"]) else "N/A"
        mean_icc = f"{row['mean_icc_ms']:.2f}" if not pd.isna(row["mean_icc_ms"]) else "N/A"
        eff = f"{row['mean_efficiency_pct']:.1f}%" if not pd.isna(row["mean_efficiency_pct"]) else "N/A"
        pv = f"{row['p_value']:.4f}" if not pd.isna(row["p_value"]) else "N/A"
        sig_marker = "*" if row["significant"] else " "
        pv_display = f"{pv}{sig_marker}"

        print(
            f"  {row['N']:<{cw['N']},} | "
            f"{churn_pct:<{cw['churn']}} | "
            f"{mean_fr:<{cw['fr']}} | "
            f"{mean_icc:<{cw['icc']}} | "
            f"{eff:<{cw['eff']}} | "
            f"{pv_display:<{cw['p']}} | "
            f"{row['h1_criterion']:<{cw['crit']}}"
        )

    print(sep)
    print(
        f"\n  (* = statistically significant at alpha=0.05)\n"
        f"\n  Files saved:\n"
        f"    {RAW_CSV}\n"
        f"    {SUMMARY_CSV}"
    )
    print()

    # -----------------------------------------------------------------------
    # H1 Overall verdict
    # -----------------------------------------------------------------------
    _banner("H1 VERDICT")
    low_churn = [
        r for r in display_rows
        if r["churn_rate"] <= 0.05 and not pd.isna(r["mean_efficiency_pct"])
    ]
    high_churn = [
        r for r in display_rows
        if r["churn_rate"] >= 0.10 and not pd.isna(r["mean_efficiency_pct"])
    ]

    low_meets = all(r["h1_criterion"] == "MEETS_TARGET" for r in low_churn)
    high_sig = any(r["significant"] for r in high_churn)
    high_meets = all(r["h1_criterion"] == "MEETS_TARGET" for r in high_churn)

    if low_meets and high_meets:
        verdict = "FULL ACCEPT  -- ICC >= 80% faster at all churn rates tested."
    elif low_meets and high_sig:
        verdict = (
            "PARTIAL ACCEPT -- ICC >= 80% faster at churn <= 5%,\n"
            "                  still significant but below 80% at churn 10%."
        )
    elif low_meets:
        verdict = (
            "PARTIAL ACCEPT -- ICC >= 80% faster at churn <= 5%,\n"
            "                  NOT significant at churn 10%."
        )
    else:
        verdict = "REJECT -- ICC does not achieve >= 80% efficiency at churn <= 5%."

    print(f"  {verdict}")
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H1 Experiment: ICC vs Full Recompute (igraph backend)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print checkpoint status (combos done/pending) and exit.",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    run_h1_experiment()
