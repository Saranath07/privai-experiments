"""
Label preprocessing: Randomized Response (LDP) + Huber corruption.

Apply ONCE before training, save to disk. Never re-run inside the training loop.
"""

import math
import numpy as np
from datasets import load_dataset, Dataset
from pathlib import Path


def c_eps(eps: float) -> float:
    if eps == float("inf") or eps > 100:
        return 1.0
    return (math.exp(eps) + 1) / (math.exp(eps) - 1)


def randomized_response(labels: np.ndarray, eps: float) -> np.ndarray:
    """
    Apply ε-LDP Randomized Response to binary labels ∈ {0,1}.
    Returns z_i ∈ {-1,+1} (sign convention used in SquareχPO loss).

    P(keep) = e^ε / (1 + e^ε).  At ε=∞: no flip.
    """
    if eps == float("inf") or eps > 100:
        return 2 * labels.astype(float) - 1

    p_keep = math.exp(eps) / (1 + math.exp(eps))
    flip_mask = np.random.binomial(1, 1 - p_keep, size=len(labels))
    noisy = labels.copy().astype(int)
    noisy[flip_mask == 1] = 1 - noisy[flip_mask == 1]
    return 2 * noisy.astype(float) - 1


def huber_corruption(labels: np.ndarray, alpha: float) -> np.ndarray:
    """α-Huber label corruption: adversarially flip α fraction of labels."""
    if alpha == 0.0:
        return labels.copy()
    n_corrupt = int(alpha * len(labels))
    idx = np.random.choice(len(labels), n_corrupt, replace=False)
    corrupted = labels.copy().astype(int)
    corrupted[idx] = 1 - corrupted[idx]
    return corrupted


def prepare_hh_rlhf(
    eps: float,
    alpha: float,
    mode: str,
    n_samples: Optional[int] = None,
    seed: int = 42,
    output_dir: str = "data",
) -> Dataset:
    """
    Load HH-RLHF, apply privacy/corruption, save to disk.

    mode: "CTL" (corrupt then LDP) | "LTC" (LDP then corrupt) | "clean" (no noise)
    Labels are always 1 (chosen response is the preferred one by construction).
    Adds 'noisy_label' column (float, ±1).
    """
    np.random.seed(seed)

    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    labels = np.ones(len(ds), dtype=int)  # chosen=1 by dataset construction

    if mode == "CTL":
        labels = huber_corruption(labels, alpha).astype(int)
        noisy = randomized_response(labels, eps)
    elif mode == "LTC":
        noisy_pm1 = randomized_response(labels, eps)
        labels_after_ldp = ((noisy_pm1 + 1) / 2).astype(int)
        labels_after_ldp = huber_corruption(labels_after_ldp, alpha).astype(int)
        noisy = 2 * labels_after_ldp.astype(float) - 1
    else:  # clean / no privacy
        noisy = 2 * labels.astype(float) - 1

    ds = ds.add_column("noisy_label", noisy.tolist())

    # Verify flip rate matches theory
    expected_flip = 0.0 if eps == float("inf") else 1 / (math.exp(eps) + 1)
    actual_flip = float((noisy != (2 * np.ones(len(ds)) - 1)).mean())
    print(f"[preprocess] ε={eps}, α={alpha}, mode={mode}: "
          f"flip_rate={actual_flip:.3f} (expected≈{expected_flip:.3f}), n={len(ds)}")

    tag = f"eps{'inf' if eps == float('inf') else eps}_alpha{alpha}_{mode}_n{len(ds)}_seed{seed}"
    save_path = Path(output_dir) / f"hh_rlhf_{tag}"
    ds.save_to_disk(str(save_path))
    print(f"[preprocess] Saved to {save_path}")
    return ds


def prepare_tldr(
    eps: float,
    alpha: float = 0.0,
    mode: str = "CTL",
    n_samples: Optional[int] = None,
    seed: int = 42,
    output_dir: str = "data",
) -> Dataset:
    """Same as prepare_hh_rlhf but for TL;DR summarization dataset."""
    np.random.seed(seed)

    ds = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    # TL;DR: choice field is 0 or 1 indicating which summary is preferred
    labels = np.array(ds["choice"], dtype=int)

    if mode == "CTL":
        labels = huber_corruption(labels, alpha).astype(int)
        noisy = randomized_response(labels, eps)
    elif mode == "LTC":
        noisy_pm1 = randomized_response(labels, eps)
        labels_after_ldp = ((noisy_pm1 + 1) / 2).astype(int)
        labels_after_ldp = huber_corruption(labels_after_ldp, alpha).astype(int)
        noisy = 2 * labels_after_ldp.astype(float) - 1
    else:
        noisy = 2 * labels.astype(float) - 1

    ds = ds.add_column("noisy_label", noisy.tolist())

    tag = f"eps{'inf' if eps == float('inf') else eps}_alpha{alpha}_{mode}_n{len(ds)}_seed{seed}"
    save_path = Path(output_dir) / f"tldr_{tag}"
    ds.save_to_disk(str(save_path))
    print(f"[preprocess] TL;DR saved to {save_path}")
    return ds


# Allow Optional import
from typing import Optional
