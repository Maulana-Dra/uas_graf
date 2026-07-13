"""
run_h2_h4_sequence.py
======================
Orchestration script to run H4 and then H2 experiments automatically and
sequentially, streaming subprocess output in real-time to both the console
and a unified log file.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Path setup — make sure experiments directory is importable
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
_FRAMEWORK_ROOT = _EXPERIMENTS_DIR.parent
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

SEQUENCE_LOG_PATH = Path("data/h2_h4_sequence_log.txt")


def log_seq(message: str) -> None:
    """Print message to stdout and append it to the sequence log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        SEQUENCE_LOG_PATH.parent.mkdir(exist_ok=True)
        with open(SEQUENCE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"  (warning: failed to write sequence log: {e})")


def run_experiment(script_path: str, name: str) -> bool:
    """Run a single experiment script as a subprocess, streaming 
    its output to both console and the sequence log. Returns True 
    if it completed successfully (exit code 0), False otherwise.
    """
    log_seq(f"=== STARTING {name} ({script_path}) ===")
    start_time = time.perf_counter()
    
    try:
        # Run as subprocess, redirecting stderr to stdout to capture everything
        # in a single unified stream
        process = subprocess.Popen(
            [sys.executable, "-u", script_path],  # Use -u to force unbuffered stdout/stderr
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            encoding="utf-8",
            errors="replace"
        )
        
        # Read subprocess stdout in real-time, line-by-line
        if process.stdout is not None:
            for line in process.stdout:
                line_stripped = line.rstrip()
                print(line_stripped)
                try:
                    # Append raw subprocess line directly to log
                    with open(SEQUENCE_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(line_stripped + "\n")
                except Exception:
                    pass  # don't let log file write failures crash the run
                    
        process.wait()
        elapsed = time.perf_counter() - start_time
        
        if process.returncode == 0:
            log_seq(f"=== FINISHED {name} SUCCESSFULLY | elapsed={elapsed/60:.1f} min ===")
            return True
        else:
            log_seq(f"=== {name} EXITED WITH ERROR CODE {process.returncode} | elapsed={elapsed/60:.1f} min ===")
            return False
            
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        log_seq(f"=== {name} CRASHED: {e} | elapsed={elapsed/60:.1f} min ===")
        return False


def main() -> None:
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    log_seq("SEQUENCE STARTED: H4 -> H2")
    sequence_start = time.perf_counter()
    
    # Step 1: Run H4 (fast, ~20-60 min expected)
    h4_success = run_experiment("experiments/run_h4.py", "H4")
    
    if not h4_success:
        log_seq(
            " H4 did not complete successfully. Stopping sequence "
            "— H2 will NOT run automatically. Check the log above "
            "for the error, fix it, then re-run this script (it "
            "will attempt H4 again since H4 has no checkpoint) or "
            "run experiments/run_h2.py manually if H4's failure is "
            "unrelated to H2."
        )
        sys.exit(1)
        
    log_seq("H4 completed. Proceeding to H2 in 10 seconds...")
    time.sleep(10)  # brief pause to allow file systems to flush
    
    # Step 2: Run H2 (slower, ~3.2 hours expected with 3 scales)
    h2_success = run_experiment("experiments/run_h2.py", "H2")
    
    total_elapsed = time.perf_counter() - sequence_start
    
    if h2_success:
        log_seq(
            f" SEQUENCE COMPLETE: Both H4 and H2 finished "
            f"successfully. Total sequence time: "
            f"{total_elapsed/3600:.2f} hours."
        )
        log_seq(
            "Results available in: data/h2_raw.csv, "
            "data/h2_summary.csv, data/h4_raw.csv, "
            "data/h4_summary.csv"
        )
    else:
        log_seq(
            f" SEQUENCE INCOMPLETE: H4 succeeded but H2 failed. "
            f"Total time before failure: {total_elapsed/3600:.2f} hours. "
            f"Check log above for H2's error, fix it, then re-run "
            f"experiments/run_h2.py directly (H4 does not need to "
            f"be re-run since it already succeeded)."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
