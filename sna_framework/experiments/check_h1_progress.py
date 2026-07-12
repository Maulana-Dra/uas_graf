"""
check_h1_progress.py
====================
Standalone script to check H1 experiment progress without interrupting
the running process. Safe to run in a separate terminal while run_h1.py
is executing in another.

This script is READ-ONLY — it never writes to h1_checkpoint.json,
h1_raw.csv, h1_summary.csv, or h1_live_log.txt.

Usage
-----
  python experiments/check_h1_progress.py   # from sna_framework/
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
DATA_DIR = os.path.join(_FRAMEWORK_ROOT, "data")

CHECKPOINT_PATH = Path(DATA_DIR) / "h1_checkpoint.json"
LIVE_LOG_PATH   = Path(DATA_DIR) / "h1_live_log.txt"
RAW_CSV_PATH    = Path(DATA_DIR) / "h1_raw.csv"

# All 9 combinations in order
ALL_COMBOS = [
    [N, c]
    for N in [10_000, 50_000, 100_000]
    for c in [0.01, 0.05, 0.10]
]


def check_progress() -> None:
    """Print checkpoint status and last 15 live-log lines."""

    print("=" * 70)
    print("H1 EXPERIMENT PROGRESS CHECK")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Checkpoint summary
    # -----------------------------------------------------------------------
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception as exc:
            print(f"[ERROR] Could not read checkpoint: {exc}")
            checkpoint = {}

        done = checkpoint.get("completed_combos", [])
        n_done = len(done)
        pending = [c for c in ALL_COMBOS if c not in done]

        print(f"\nCheckpoint: {CHECKPOINT_PATH}")
        print(f"  FR backend  : {checkpoint.get('backend', 'unknown')}")
        print(f"  ICC backend : {checkpoint.get('icc_backend', 'unknown')}")
        print(f"  Batches/combo: {checkpoint.get('n_batches_per_combo', '?')}")
        print(f"  Last updated: {checkpoint.get('last_updated', '?')}")
        print(f"\n  Completed ({n_done}/9):")
        for combo in done:
            print(f"    [OK] N={combo[0]:>7,}  churn={combo[1]*100:.0f}%")
        if pending:
            print(f"\n  Pending ({len(pending)}/9):")
            for combo in pending:
                print(f"    [ ]  N={combo[0]:>7,}  churn={combo[1]*100:.0f}%")
        else:
            print("\n  All 9 combinations complete!")
    else:
        print(f"\nNo checkpoint found at: {CHECKPOINT_PATH}")
        print("  Experiment may not have completed any combination yet,")
        print("  or data/h1_checkpoint.json was deleted.")

    # -----------------------------------------------------------------------
    # Raw CSV row count
    # -----------------------------------------------------------------------
    if RAW_CSV_PATH.exists():
        try:
            with open(RAW_CSV_PATH, "r", encoding="utf-8") as f:
                n_rows = sum(1 for _ in f) - 1  # subtract header
            print(f"\n  h1_raw.csv: {n_rows} data rows written so far (max 135)")
        except Exception as exc:
            print(f"\n  [WARN] Could not count rows in h1_raw.csv: {exc}")
    else:
        print(f"\n  h1_raw.csv: not yet created")

    # -----------------------------------------------------------------------
    # Live log tail
    # -----------------------------------------------------------------------
    print()
    if LIVE_LOG_PATH.exists():
        try:
            with open(LIVE_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            n_lines = len(lines)
            tail = lines[-15:] if n_lines >= 15 else lines
            print(f"Live log: {LIVE_LOG_PATH} ({n_lines} lines total)")
            print(f"Last {len(tail)} lines:")
            print("-" * 70)
            for line in tail:
                print(line.rstrip())
        except Exception as exc:
            print(f"[ERROR] Could not read live log: {exc}")
    else:
        print(f"No live log found at: {LIVE_LOG_PATH}")
        print("  run_h1.py has not started yet, or the log file was not created.")

    print()


if __name__ == "__main__":
    check_progress()
