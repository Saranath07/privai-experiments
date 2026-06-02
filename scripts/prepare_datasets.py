"""
Build all noisy datasets for the full experiment grid and save to disk.

Run ONCE on the GPU machine before any training:
    python scripts/prepare_datasets.py --output_dir data/

This ensures every training run uses identical pre-flipped labels.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocess_labels import prepare_hh_rlhf, prepare_tldr

EPS_VALUES   = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
ALPHA_VALUES = [0.0, 0.05, 0.1]
MODES        = ["CTL", "LTC"]
N_SCALING    = [10_000, 25_000, 50_000, 100_000]   # for n-scaling ablation
SEED         = 42


def main(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 1: Full HH-RLHF grid (all ε × α × mode)")
    print("=" * 60)
    for eps in EPS_VALUES:
        for alpha in ALPHA_VALUES:
            for mode in MODES:
                if mode == "LTC" and alpha == 0.0:
                    # CTL and LTC are identical when α=0 — skip duplicates
                    continue
                prepare_hh_rlhf(eps=eps, alpha=alpha, mode=mode,
                                 seed=SEED, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("Phase 2: HH-RLHF n-scaling subsets (ε ∈ {0.5,1,2}, CTL, α=0)")
    print("=" * 60)
    for eps in [0.5, 1.0, 2.0]:
        for n in N_SCALING:
            prepare_hh_rlhf(eps=eps, alpha=0.0, mode="CTL",
                             n_samples=n, seed=SEED, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("Phase 3: TL;DR (ε ∈ {0.5,1,2,∞}, CTL, α=0)")
    print("=" * 60)
    for eps in [0.5, 1.0, 2.0, float("inf")]:
        prepare_tldr(eps=eps, alpha=0.0, mode="CTL",
                     seed=SEED, output_dir=output_dir)

    print("\nAll datasets prepared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data", help="Directory to save datasets")
    args = parser.parse_args()
    main(args.output_dir)
