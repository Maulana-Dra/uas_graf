"""
check_h2_h4_progress.py
=======================
Standalone read-only progress checker for the H4->H2 sequence.
Safe to run in a separate terminal while run_h2_h4_sequence.py is executing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Path setup
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _EXPERIMENTS_DIR.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))


def check_progress() -> None:
    log_path = Path("data/h2_h4_sequence_log.txt")
    h4_summary = Path("data/h4_summary.csv")
    h2_summary = Path("data/h2_summary.csv")
    
    print("=" * 70)
    print("H4 -> H2 SEQUENCE PROGRESS CHECK")
    print("=" * 70)
    
    print(f"\nH4 summary exists: {h4_summary.exists()}")
    if h4_summary.exists():
        try:
            import pandas as pd
            df = pd.read_csv(h4_summary)
            print(f"  Rows in h4_summary.csv: {len(df)}")
        except Exception as e:
            print(f"  (error reading h4_summary.csv: {e})")
            
    print(f"\nH2 summary exists: {h2_summary.exists()}")
    if h2_summary.exists():
        try:
            import pandas as pd
            df = pd.read_csv(h2_summary)
            print(f"  Rows in h2_summary.csv: {len(df)}")
        except Exception as e:
            print(f"  (error reading h2_summary.csv: {e})")
            
    print()
    if log_path.exists():
        print("Last 20 lines of sequence log:")
        print("-" * 70)
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())
        except Exception as e:
            print(f"  (error reading sequence log file: {e})")
    else:
        print("No sequence log found yet — sequence may not have started.")


if __name__ == "__main__":
    check_progress()
