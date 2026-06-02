"""
Launch the full experiment grid sequentially on the GPU machine.

Usage:
    python scripts/launch_grid.py --phase sanity    # 3 runs, quick check
    python scripts/launch_grid.py --phase ablation  # full 4B grid
    python scripts/launch_grid.py --phase flagship  # 12B best configs
    python scripts/launch_grid.py --phase nscaling  # dataset size ablation

Runs are launched sequentially (not parallel) since we have 1-2 GPUs.
Each run logs to results/{run_name}.json.
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_CMD = [sys.executable, "scripts/train.py"]


def run(extra_args: list[str], dry_run: bool = False):
    cmd = BASE_CMD + extra_args + ["--wandb"]
    print("CMD:", " ".join(cmd))
    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[WARNING] Run failed: {' '.join(extra_args)}")


def phase_sanity(dry_run):
    """3 runs to verify the implementation before committing to the full grid."""
    print("\n--- SANITY PHASE (3 runs) ---")
    # 1. Clean baseline
    run(["--eps", "inf", "--alpha", "0.0", "--mode", "CTL",
         "--model_size", "4b", "--n_samples", "10000"], dry_run)
    # 2. LLDP CTL ε=1.0
    run(["--eps", "1.0", "--alpha", "0.0", "--mode", "CTL",
         "--model_size", "4b", "--n_samples", "10000"], dry_run)
    # 3. LLDP LTC ε=1.0 (should be worse than CTL)
    run(["--eps", "1.0", "--alpha", "0.0", "--mode", "LTC",
         "--model_size", "4b", "--n_samples", "10000"], dry_run)


def phase_ablation(dry_run):
    """Full 4B ablation grid."""
    print("\n--- ABLATION PHASE (4B, full grid) ---")
    EPS    = ["inf", "8.0", "4.0", "2.0", "1.0", "0.5", "0.1"]
    ALPHAS = ["0.0", "0.05", "0.1"]
    MODES  = ["CTL", "LTC"]

    for eps in EPS:
        # Privacy-only runs (α=0): both CTL and LTC
        for mode in MODES:
            run(["--eps", eps, "--alpha", "0.0", "--mode", mode,
                 "--model_size", "4b"], dry_run)
        # Privacy + corruption: only at ε ∈ {0.5, 1.0, 2.0}
        if eps in ["0.5", "1.0", "2.0"]:
            for alpha in ["0.05", "0.1"]:
                for mode in MODES:
                    run(["--eps", eps, "--alpha", alpha, "--mode", mode,
                         "--model_size", "4b"], dry_run)


def phase_flagship(dry_run):
    """Best configs on 12B."""
    print("\n--- FLAGSHIP PHASE (12B) ---")
    configs = [
        ("inf",  "0.0", "CTL"),
        ("2.0",  "0.0", "CTL"),
        ("1.0",  "0.0", "CTL"),
        ("0.5",  "0.0", "CTL"),
        ("1.0",  "0.0", "LTC"),
        ("1.0",  "0.1", "CTL"),
    ]
    for eps, alpha, mode in configs:
        run(["--eps", eps, "--alpha", alpha, "--mode", mode,
             "--model_size", "12b"], dry_run)


def phase_nscaling(dry_run):
    """Dataset size ablation."""
    print("\n--- N-SCALING PHASE ---")
    N_SIZES = ["10000", "25000", "50000", "100000"]
    for eps in ["0.5", "1.0", "2.0"]:
        for n in N_SIZES:
            run(["--eps", eps, "--alpha", "0.0", "--mode", "CTL",
                 "--model_size", "4b", "--n_samples", n], dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["sanity", "ablation", "flagship", "nscaling", "all"])
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    if args.phase == "sanity"   or args.phase == "all": phase_sanity(args.dry_run)
    if args.phase == "ablation" or args.phase == "all": phase_ablation(args.dry_run)
    if args.phase == "flagship" or args.phase == "all": phase_flagship(args.dry_run)
    if args.phase == "nscaling" or args.phase == "all": phase_nscaling(args.dry_run)
