"""
run_h1.py
=========
H1 Experiment: ICC vs Full Recompute speed comparison on dynamic BA graphs.

=============================================================================
METHODOLOGICAL ADJUSTMENT FOR COMPUTE CONSTRAINTS (N=100,000 COMBINATIONS)

PATH A (Recommended if there is any doubt about timing):
  Let the current combination finish naturally. The moment it finishes (before
  N=100,000 churn=0.01 begins), the running process will hit the checkpoint
  save point. AT THAT EXACT MOMENT, if this code has already been updated on
  disk, Ctrl+C immediately after seeing the "FINISHED N=50000" log line but
  BEFORE the next combo's tqdm bar appears, then restart with:
    python experiments/run_h1.py
  It will resume correctly (skip the completed combos) and apply the new
  override for N=100,000.

PATH B (Safer, avoids precise timing):
  Simply let the current process run using the old code (all 15 batches) for the
  combo currently in flight. Apply the code changes to disk now. The next time the
  script is restarted (by manual Ctrl+C or if it happens to crash/pause), it will
  pick up the new override starting from whichever combo hasn't been checkpointed
  yet.
=============================================================================

# =============================================================================
# BEFORE RE-RUNNING:
# 1. Delete these files completely (ALL prior data must be regenerated for
#    methodological consistency — both FR and ICC now use igraph backend):
#      data/h1_raw.csv
#      data/h1_summary.csv
#      data/h1_checkpoint.json
# 2. Run: pip install python-igraph   (if not already done)
# 3. Run: python experiments/run_h1.py
#    (N=10,000 will finish in minutes with the new igraph-accelerated ICC)
# 4. You can safely Ctrl+C between combinations; progress is saved after
#    each completed (N, churn_rate) combo. Resume by simply running the
#    same command again.
# 5. Check progress anytime with: python experiments/run_h1.py --status
# =============================================================================

Hypothesis H1
-------------
Incremental Closeness Centrality (ICC) is >= 80% faster than Full Recompute
(FR) on dynamic Barabasi-Albert graphs with churn rate < 10%.

Acceptance criteria
-------------------
  FULL ACCEPT   : mean efficiency_pct >= 80% at churn 1% AND 5%
  PARTIAL ACCEPT: mean efficiency_pct >= 80% only at churn < 5%,
                  still statistically significant at churn 10%
  REJECT        : ICC slower or not statistically significant at churn 10%

Statistical test
----------------
  Normality check on paired differences via Shapiro-Wilk (alpha=0.05).
  - If normal  -> Paired T-Test  (scipy.stats.ttest_rel)
  - If not     -> Wilcoxon Signed-Rank (scipy.stats.wilcoxon)

Experimental design
-------------------
  Graph sizes  : 10_000 / 50_000 / 100_000 nodes
  Churn rates  : 0.01 / 0.05 / 0.10
  Batches/combo: 15 measured + 1 warm-up (discarded)
  BA parameters: m=2 (avg degree ~4), seed=42
  Batch seeds  : 42 + batch_index  (reproducible, different per batch)
  FR Backend   : igraph (C-level) for FullRecompute baseline
  ICC Backend  : igraph batched distances() for per-affected-node BFS

Output files
------------
  data/h1_raw.csv        – one row per measured batch (135 rows max)
  data/h1_summary.csv    – one row per (N, churn) combo (9 rows)
  data/h1_checkpoint.json – tracks completed combos for resume support

Usage
-----
  python experiments/run_h1.py           # from sna_framework/
  python experiments/run_h1.py --status  # show checkpoint progress only
  cd experiments && python run_h1.py     # from experiments/
"""

from __future__ import annotations

import sys
import os
import json
import warnings
import datetime
import argparse

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
N_BATCHES_DEFAULT = 15  # Standard batch count (Wilcoxon/Paired T-Test minimum recommended n>=15)

# N_BATCHES_OVERRIDE was previously used to reduce batch count 
# for N=100,000 churn=0.05/0.10 to 5 batches each, due to 
# projected excessive runtime (~20-30+ hours per combination at 
# full 15 batches). This override has been REMOVED per updated 
# instruction from the course instructor (Dr. Dian Puspita 
# Hapsari), who granted a schedule extension on the condition 
# that all 9 combinations use the full, methodologically 
# consistent 15-batch standard. As of 2026-07-12, all 
# combinations use N_BATCHES_DEFAULT (15).
N_BATCHES_OVERRIDE = {}

def get_n_batches(N: int, churn_rate: float) -> int:
    """Return the batch count for a given (N, churn_rate) combo,
    using the override if present, else the default.
    """
    return N_BATCHES_OVERRIDE.get((N, churn_rate), N_BATCHES_DEFAULT)


def serialize_override(override_dict: dict) -> dict[str, int]:
    """Serialize override keys as strings for JSON compatibility."""
    return {f"{k[0]}_{k[1]}": v for k, v in override_dict.items()}


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
    """Classify a single (N, churn) result against the H1 criterion.

    Parameters
    ----------
    efficiency_pct : float
        Mean efficiency percentage; (1 - icc_ms / fr_ms) * 100.
    churn_rate : float
        The churn rate for this experiment combo.

    Returns
    -------
    str
        One of ``"MEETS_TARGET"``, ``"BELOW_TARGET"``, or ``"ICC_SLOWER"``.
    """
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
    """Context manager that redirects stdout to /dev/null during measurements.

    Used to silence per-batch engine print() calls (which would break the
    tqdm progress bar display) while still allowing tqdm to write to stdout.
    """

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
    """Generate a batch of edge changes guaranteed to keep G connected.

    Unlike :meth:`GraphGenerator.generate_batch_updates`, this function
    probes removals **cumulatively** on an accumulating working copy of G,
    so no combination of simultaneous removals can disconnect the graph.
    Edge additions are identical to the engine implementation.

    Parameters
    ----------
    G : nx.Graph
        Current graph (read-only).
    churn_rate : float
        Fraction of current edge count to modify.
    add_fraction : float, optional
        Fraction of changes that are additions (default 0.80).
    seed : int or None, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{"added": [(u, v), ...], "removed": [(u, v), ...]}``
        Applying this dict to G is guaranteed to leave G connected.
    """
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

    # ---- Edge removals — cumulative probing on a working copy -------------
    # Each candidate removal is tested on the ALREADY-REDUCED working copy,
    # so the combined set of removals is guaranteed to keep connectivity.
    removed: list[tuple[int, int]] = []
    G_work = G.copy()            # accumulates removals as they are accepted
    candidate_edges = list(G_work.edges())
    rng.shuffle(candidate_edges)

    for edge in candidate_edges:
        if len(removed) >= n_remove:
            break
        u, v = edge
        # Skip if removing would isolate either endpoint
        if G_work.degree(u) <= 1 or G_work.degree(v) <= 1:
            continue
        # Test connectivity on the CURRENT working copy (not original G)
        G_work.remove_edge(u, v)
        if nx.is_connected(G_work):
            removed.append((u, v))
            # Leave the working copy without this edge for subsequent probes
        else:
            G_work.add_edge(u, v)   # restore: this removal was unsafe

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


def _save_checkpoint(completed_combos: list[list]) -> None:
    """Write/update checkpoint file with current completed combos."""
    checkpoint = {
        "completed_combos": completed_combos,
        "backend": CHECKPOINT_BACKEND,
        "icc_backend": CHECKPOINT_ICC_BACKEND,
        "n_batches_per_combo": N_BATCHES_DEFAULT,
        "n_batches_override": serialize_override(N_BATCHES_OVERRIDE),
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def _append_raw_rows(raw_rows: list[dict], raw_columns: list[str]) -> None:
    """Append raw batch rows to h1_raw.csv (write header only if new file)."""
    df = pd.DataFrame(raw_rows, columns=raw_columns)
    write_header = not os.path.exists(RAW_CSV)
    df.to_csv(RAW_CSV, mode="a", index=False, header=write_header)


def _append_summary_row(summary_row: dict, summary_columns: list[str]) -> None:
    """Append one summary row to h1_summary.csv (write header only if new)."""
    df = pd.DataFrame([summary_row], columns=summary_columns)
    write_header = not os.path.exists(SUMMARY_CSV)
    df.to_csv(SUMMARY_CSV, mode="a", index=False, header=write_header)


# ---------------------------------------------------------------------------
# --status command
# ---------------------------------------------------------------------------

def print_status() -> None:
    """Print checkpoint contents and pending combos, then exit."""
    checkpoint = _load_checkpoint()
    all_combos = [[N, c] for N in GRAPH_SIZES for c in CHURN_RATES]
    total = len(all_combos)

    print("\n=== H1 Experiment Status ===")
    if checkpoint is None:
        print(f"  No checkpoint found at: {CHECKPOINT_JSON}")
        print(f"  0 combos done, {total} pending.")
    else:
        done = checkpoint.get("completed_combos", [])
        n_done = len(done)
        pending = [c for c in all_combos if c not in done]
        print(f"  Checkpoint  : {CHECKPOINT_JSON}")
        print(f"  FR Backend  : {checkpoint.get('backend', 'unknown')}")
        print(f"  ICC Backend : {checkpoint.get('icc_backend', 'unknown (old checkpoint)')}")
        print(f"  Batches     : {checkpoint.get('n_batches_per_combo', '?')}")
        print(f"  Overrides   : {checkpoint.get('n_batches_override', 'None')}")
        print(f"  Updated     : {checkpoint.get('last_updated', '?')}")
        print(f"\n  Completed ({n_done}/{total}):")
        for combo in done:
            print(f"    N={combo[0]:>7,}  churn={combo[1]*100:.0f}%")
        print(f"\n  Pending ({len(pending)}/{total}):")
        for combo in pending:
            print(f"    N={combo[0]:>7,}  churn={combo[1]*100:.0f}%")
    print()


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_h1_experiment() -> None:
    """Run the full H1 experiment across all (N, churn) combinations.

    Generates BA graphs, measures FR (igraph backend) and ICC timing over
    15 batches per combination, performs statistical tests, and saves results
    to CSV files. Progress is checkpointed after each completed combo, so
    the run can be safely interrupted and resumed.
    Prints a formatted summary table at the end.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    session_start = datetime.datetime.now()

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
            "  This likely means the old NetworkX-backend data is present "
            "without a checkpoint.\n"
            "  Mixing old NetworkX-backend data with new igraph-backend "
            "data would confound the results.\n"
            "  Please manually delete these files before running:\n"
            f"    {RAW_CSV}\n"
            f"    {SUMMARY_CSV}\n"
        )
        sys.exit(1)

    completed_combos: list[list] = []
    if checkpoint is not None:
        completed_combos = checkpoint.get("completed_combos", [])
        ckpt_backend = checkpoint.get("backend")
        ckpt_icc_backend = checkpoint.get("icc_backend")
        ckpt_n_batches = checkpoint.get("n_batches_per_combo")
        
        # Check if the checkpoint's override scheme matches the current one.
        ckpt_override = checkpoint.get("n_batches_override")
        current_override_serialized = serialize_override(N_BATCHES_OVERRIDE)
        
        mismatch_override = False
        if ckpt_override is not None:
            if ckpt_override != current_override_serialized:
                mismatch_override = True
        else:
            # If missing, it's a mismatch if any completed combo should have been overridden
            for combo in completed_combos:
                if tuple(combo) in N_BATCHES_OVERRIDE:
                    mismatch_override = True

        # Check for core mismatches (backend or standard batch count)
        core_mismatch = (
            ckpt_backend != CHECKPOINT_BACKEND
            or ckpt_icc_backend != CHECKPOINT_ICC_BACKEND
            or ckpt_n_batches != N_BATCHES_DEFAULT
        )

        if core_mismatch:
            print(
                "\n[ERROR] Checkpoint was created with a different backend or standard batch count.\n"
                "  Checkpoint FR backend  : "
                f"{ckpt_backend!r}  (expected: {CHECKPOINT_BACKEND!r})\n"
                "  Checkpoint ICC backend : "
                f"{ckpt_icc_backend!r}  (expected: {CHECKPOINT_ICC_BACKEND!r})\n"
                "  Checkpoint batches     : "
                f"{ckpt_n_batches}  (expected: {N_BATCHES_DEFAULT})\n"
                "  Delete data/h1_raw.csv, data/h1_summary.csv, and data/h1_checkpoint.json to start fresh.\n"
            )
            sys.exit(1)

        if mismatch_override:
            # Auto-update the override scheme in the checkpoint since no completed combo data is affected
            # (completed combos 1-6 did not use any overrides under either the old or new scheme)
            log_live(
                "Note: batch override scheme changed (previously reduced N=100,000 high-churn combos to 5 batches, "
                "now using standard 15 for all). No completed data is affected. Checkpoint updated."
            )
            checkpoint["n_batches_override"] = current_override_serialized
            # Save updated checkpoint back to disk
            with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
            mismatch_override = False

        n_done = len(completed_combos)
        if n_done > 0:
            print(
                f"\n  Resuming: {n_done} combos already done with "
                f"igraph FR+ICC backend, skipping them."
            )

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

    # Column definitions (shared across append helpers)
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

    # Log overall run start (after resume guard so completed_combos is known)
    all_combos = [[N, c] for N in GRAPH_SIZES for c in CHURN_RATES]
    log_live("H1 EXPERIMENT STARTED (or resumed)")
    log_live(f"Combos already completed (skipped): {completed_combos}")
    remaining_combos = [c for c in all_combos if c not in completed_combos]
    log_live(f"Combos remaining: {remaining_combos}")

    # -----------------------------------------------------------------------
    # Outer loops: N × churn_rate
    # -----------------------------------------------------------------------
    for N in GRAPH_SIZES:
        for churn_rate in CHURN_RATES:
            combo_key = [N, churn_rate]

            # Skip already-completed combos
            if combo_key in completed_combos:
                print(
                    f"\n  [SKIP] N={N:,} churn={churn_rate*100:.0f}% "
                    f"— already completed (igraph backend)."
                )
                continue

            n_batches_this_combo = get_n_batches(N, churn_rate)
            is_reduced = (N, churn_rate) in N_BATCHES_OVERRIDE
            batch_note = (
                f"REDUCED ({n_batches_this_combo} batches, standard is "
                f"{N_BATCHES_DEFAULT}) — see BAB II metodologi for rationale"
                if is_reduced else
                f"standard ({n_batches_this_combo} batches)"
            )

            _banner(f"N={N:,}  |  churn_rate={churn_rate*100:.0f}%")
            log_live(f"=== STARTING N={N} churn={churn_rate} | {batch_note} ===")

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
                # Step 2: Initialise ICC (initialization timing NOT measured)
                # ------------------------------------------------------------
                print(f"  Initialising ICC ...")
                # Pass a copy so ICC's internal .G is independent of our main G
                icc = IncrementalCloseness(G.copy())

                # ------------------------------------------------------------
                # Step 3a: Warm-up batch (discard timing)
                # ------------------------------------------------------------
                print(f"  Running {N_WARMUP} warm-up batch(es) [not measured] ...")
                for wu in range(N_WARMUP):
                    warmup_seed = BASE_SEED - 1 - wu
                    wu_batch = _safe_batch(
                        G, churn_rate=churn_rate, seed=warmup_seed
                    )
                    # Advance main G through the warm-up batch
                    with _SuppressPrint():
                        gen.apply_batch(G, wu_batch)

                    # Throwaway FR call for JIT warm-up (time the code path)
                    G_wu = G.copy()
                    with _SuppressPrint():
                        gen.apply_batch(G_wu, wu_batch)
                    fr_engine.compute(G_wu)
                    del G_wu

                # Re-initialise ICC on the post-warmup G so it starts in sync
                print(f"  Re-initialising ICC on post-warmup graph ...")
                icc = IncrementalCloseness(G.copy())

                # ------------------------------------------------------------
                # Step 3b: measured batches
                # ------------------------------------------------------------
                raw_rows: list[dict] = []
                skipped: int = 0
                batch_times_this_combo: list[float] = []  # (fr_s + icc_s) per batch

                pbar_label = f"  N={N:>7,} churn={churn_rate*100:.0f}%"
                with tqdm(
                    total=n_batches_this_combo,
                    desc=pbar_label,
                    unit="batch",
                    ncols=80,
                    file=sys.stderr,   # tqdm to stderr so stdout suppression doesn't kill it
                ) as pbar:

                    for i in range(n_batches_this_combo):
                        batch_seed = BASE_SEED + i

                        # ---- a. Generate safe batch plan ------------------
                        batch = _safe_batch(
                            G, churn_rate=churn_rate, seed=batch_seed
                        )

                        # ---- b. Measure FR ---------------------------------
                        G_fr = G.copy()
                        with _SuppressPrint():
                            gen.apply_batch(G_fr, batch)
                        # Time ONLY the compute() call
                        fr_result = fr_engine.compute(G_fr)
                        fr_ms: float = fr_result["elapsed_ms"]
                        del G_fr

                        # ---- c. Measure ICC --------------------------------
                        icc_result = icc.update(batch)
                        icc_ms: float = icc_result["elapsed_ms"]
                        n_affected: int = icc_result["n_affected"]

                        # ---- d. Advance main reference G -------------------
                        with _SuppressPrint():
                            gen.apply_batch(G, batch)

                        # ---- Guard: zero ICC timing ------------------------
                        if icc_ms == 0.0:
                            warnings.warn(
                                f"icc_elapsed_ms==0 at batch {i} "
                                f"(N={N}, churn={churn_rate}); "
                                "setting speedup=inf, efficiency=100.0",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            speedup_ratio = float("inf")
                            efficiency_pct = 100.0
                        else:
                            speedup_ratio = fr_ms / icc_ms
                            efficiency_pct = (1.0 - icc_ms / fr_ms) * 100.0

                        # ---- e. Collect row --------------------------------
                        raw_rows.append({
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
                            "n_batches_used": n_batches_this_combo,
                            "is_reduced_batch_count": is_reduced,
                        })

                        # ---- f. Per-batch live logging (after measurement) --
                        batch_total_s = (fr_ms + icc_ms) / 1000
                        batch_times_this_combo.append(batch_total_s)

                        avg_batch_s = (
                            sum(batch_times_this_combo)
                            / len(batch_times_this_combo)
                        )
                        batches_remaining = (
                            n_batches_this_combo - len(batch_times_this_combo)
                        )
                        est_remaining_s = avg_batch_s * batches_remaining
                        n_affected_pct = (
                            (n_affected / N * 100) if N > 0 else 0
                        )

                        log_live(
                            f"N={N} churn={churn_rate} "
                            f"batch={len(batch_times_this_combo)}/{n_batches_this_combo} | "
                            f"FR={fr_ms/1000:.1f}s | "
                            f"ICC={icc_ms/1000:.1f}s "
                            f"(n_affected={n_affected:,}/{N:,}={n_affected_pct:.1f}%) | "
                            f"batch_total={batch_total_s:.1f}s | "
                            f"est. sisa kombinasi ini: "
                            f"~{est_remaining_s/3600:.2f} jam"
                        )

                        # ---- g. Anomaly warning (>=3 batches for baseline) --
                        if len(batch_times_this_combo) >= 3:
                            prev_batches = batch_times_this_combo[:-1]
                            prev_avg = sum(prev_batches) / len(prev_batches)
                            if prev_avg > 0 and batch_total_s > prev_avg * 1.5:
                                log_live(
                                    f"  [WARN] Batch ini {batch_total_s:.1f}s, "
                                    f"{(batch_total_s/prev_avg - 1)*100:.0f}% "
                                    f"lebih lama dari rata-rata batch sebelumnya "
                                    f"({prev_avg:.1f}s). Kemungkinan: thermal "
                                    f"throttling, memory pressure, atau n_affected "
                                    f"unusually high. Pantau jika berulang."
                                )

                        pbar.update(1)

                # End of batch loop
                n_valid = len(raw_rows)
                print(f"\n  Collected {n_valid} valid rows ({skipped} skipped).")

                # ------------------------------------------------------------
                # Step 4: Statistical test for this (N, churn) combo
                # ------------------------------------------------------------
                if n_batches_this_combo < 6:
                    log_live(
                        f"  ⚠️ WARNING: n={n_batches_this_combo} may be too "
                        "small for reliable Wilcoxon/Paired T-Test results. "
                        "Statistical test will still run but should be "
                        "interpreted with caution in BAB II."
                    )

                min_valid = n_batches_this_combo if n_batches_this_combo < MIN_VALID_ROWS else MIN_VALID_ROWS
                if n_valid < min_valid:
                    print(
                        f"  WARNING: only {n_valid} valid batches for "
                        f"N={N} churn={churn_rate} — marking INSUFFICIENT_DATA."
                    )
                    summary_row = {
                        "N": N,
                        "churn_rate": churn_rate,
                        "mean_fr_ms": float("nan"),
                        "mean_icc_ms": float("nan"),
                        "mean_speedup": float("nan"),
                        "mean_efficiency_pct": float("nan"),
                        "std_efficiency_pct": float("nan"),
                        "test_used": "N/A",
                        "test_statistic": float("nan"),
                        "p_value": float("nan"),
                        "significant": False,
                        "h1_criterion": "INSUFFICIENT_DATA",
                        "n_batches_used": n_batches_this_combo,
                        "is_reduced_batch_count": is_reduced,
                        "notes": "Insufficient data rows collected.",
                    }
                else:
                    fr_times = np.array([r["fr_elapsed_ms"] for r in raw_rows])
                    icc_times = np.array([r["icc_elapsed_ms"] for r in raw_rows])
                    speedup_ratios = np.array([r["speedup_ratio"] for r in raw_rows])
                    efficiency_pcts = np.array([r["efficiency_pct"] for r in raw_rows])

                    # Normality check on paired differences
                    p_shapiro = float("nan")
                    test_name = "N/A"
                    test_stat = float("nan")
                    p_value = float("nan")
                    significant = False
                    
                    try:
                        differences = fr_times - icc_times
                        shapiro_stat, p_shapiro = scipy_stats.shapiro(differences)

                        if p_shapiro >= 0.05:
                            # Differences are normally distributed -> Paired T-Test
                            test_name = "Paired T-Test"
                            test_stat, p_value = scipy_stats.ttest_rel(fr_times, icc_times)
                        else:
                            # Not normal -> Wilcoxon Signed-Rank
                            test_name = "Wilcoxon Signed-Rank"
                            # alternative='greater' tests fr > icc (ICC is faster)
                            test_stat, p_value = scipy_stats.wilcoxon(
                                fr_times, icc_times, alternative="greater"
                            )
                        significant = bool(p_value < 0.05)
                    except Exception as stats_exc:
                        log_live(f"  [WARN] Uji statistik gagal dijalankan ({stats_exc}). Fallback digunakan.")
                        test_name = "INSUFFICIENT_SAMPLE_SIZE"
                        test_stat = float("nan")
                        p_value = float("nan")
                        significant = False

                    mean_eff = float(np.mean(efficiency_pcts))

                    if is_reduced and n_batches_this_combo < 15:
                        notes_val = (
                            f"Reduced sample size (n={n_batches_this_combo}) due to "
                            "compute time constraints; statistical power is lower than "
                            "standard combinations (n=15)."
                        )
                    else:
                        notes_val = ""

                    shapiro_str = f"{p_shapiro:.4f}" if not np.isnan(p_shapiro) else "N/A"
                    test_stat_str = f"{test_stat:.4f}" if not np.isnan(test_stat) else "N/A"
                    p_val_str = f"{p_value:.6f}" if not np.isnan(p_value) else "N/A"
                    print(
                        f"  Shapiro-Wilk p={shapiro_str} -> using {test_name}\n"
                        f"  test_stat={test_stat_str}, p_value={p_val_str}, "
                        f"significant={significant}\n"
                        f"  mean_efficiency_pct={mean_eff:.2f}%  "
                        f"mean_speedup={np.mean(speedup_ratios):.3f}x"
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
                        "n_batches_used": n_batches_this_combo,
                        "is_reduced_batch_count": is_reduced,
                        "notes": notes_val,
                    }

                all_summary_rows.append(summary_row)

                # ---- Combo completion log ----------------------------------
                combo_total_s = sum(batch_times_this_combo)
                combo_avg_s = (
                    combo_total_s / len(batch_times_this_combo)
                    if batch_times_this_combo else 0.0
                )
                _mean_eff = summary_row.get("mean_efficiency_pct", float("nan"))
                _mean_eff_str = (
                    f"{_mean_eff:.1f}%"
                    if _mean_eff == _mean_eff  # NaN check
                    else "N/A"
                )
                log_live(
                    f"=== FINISHED N={N} churn={churn_rate} | "
                    f"total={combo_total_s/3600:.2f} jam | "
                    f"avg_batch={combo_avg_s:.1f}s | "
                    f"mean_efficiency_pct={_mean_eff_str} ==="
                )

                # ------------------------------------------------------------
                # Checkpoint: save combo results immediately after completion
                # ------------------------------------------------------------
                _append_raw_rows(raw_rows, raw_columns)
                _append_summary_row(summary_row, summary_columns)
                completed_combos.append(combo_key)
                _save_checkpoint(completed_combos)
                print(
                    f"  [SAVED] N={N:,} churn={churn_rate*100:.0f}% -> "
                    f"appended to CSVs; checkpoint updated "
                    f"({len(completed_combos)}/9 combos done)."
                )

            except KeyboardInterrupt:
                print(
                    f"\n\n  [INTERRUPTED] During N={N:,} churn={churn_rate*100:.0f}%.\n"
                    f"  This combo will be redone on next run (partial data NOT saved).\n"
                    f"  Progress so far: {len(completed_combos)}/9 combos saved.\n"
                )
                sys.exit(0)

            except Exception as exc:
                print(
                    f"\n  [ERROR] Exception during N={N:,} "
                    f"churn={churn_rate*100:.0f}%:\n  {exc}\n"
                    f"  Partial data NOT saved. Progress: "
                    f"{len(completed_combos)}/9 combos saved.\n"
                )
                raise

    # -----------------------------------------------------------------------
    # Session completion log
    # -----------------------------------------------------------------------
    session_elapsed = datetime.datetime.now() - session_start
    session_h = session_elapsed.total_seconds() / 3600
    log_live("=== ALL H1 COMBINATIONS COMPLETE ===")
    log_live(
        f"Total wall time this session: "
        f"{session_elapsed} (~{session_h:.2f} jam)"
    )

    # -----------------------------------------------------------------------
    # Final console summary table
    # -----------------------------------------------------------------------
    _banner("H1 EXPERIMENT RESULTS")

    # Re-load all summary rows from CSV for consistent display
    # (handles resume case where some rows were loaded from previous runs)
    if os.path.exists(SUMMARY_CSV):
        df_summary_all = pd.read_csv(SUMMARY_CSV)
        display_rows = df_summary_all.to_dict("records")
    else:
        display_rows = all_summary_rows

    # Column widths
    cw = {
        "N": 8, "churn": 6, "fr": 14, "icc": 14,
        "eff": 8, "p": 9, "crit": 18,
    }
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
        mean_fr = (
            f"{row['mean_fr_ms']:.2f}"
            if not pd.isna(row["mean_fr_ms"]) else "N/A"
        )
        mean_icc = (
            f"{row['mean_icc_ms']:.2f}"
            if not pd.isna(row["mean_icc_ms"]) else "N/A"
        )
        eff = (
            f"{row['mean_efficiency_pct']:.1f}%"
            if not pd.isna(row["mean_efficiency_pct"]) else "N/A"
        )
        pv = (
            f"{row['p_value']:.4f}"
            if not pd.isna(row["p_value"]) else "N/A"
        )
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
