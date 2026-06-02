"""
ε-adaptive training hyperparameters.

At small ε, gradient variance is amplified by c(ε)². This module scales
learning rate and gradient accumulation to compensate, keeping the effective
per-step signal-to-noise ratio roughly constant across privacy levels.
"""

import math


def c_eps(eps: float) -> float:
    if eps == float("inf") or eps > 100:
        return 1.0
    return (math.exp(eps) + 1) / (math.exp(eps) - 1)


def get_training_config(eps: float, model_size: str = "4b") -> dict:
    """
    Returns training hyperparameters adapted to the privacy budget ε.

    Scaling rules:
      LR         ∝ 1 / c(ε)         — compensates for inflated gradient scale
      grad_accum ∝ c(ε)²  (capped)  — reduces variance toward clean-DPO level
      warmup     increases with flip rate — stable direction before committing
    """
    # A100 80GB: can fit much larger batches than originally estimated
    base_lr    = {"4b": 5e-7, "12b": 2e-7}.get(model_size, 5e-7)
    base_accum = {"4b": 2,    "12b": 4   }.get(model_size, 2)   # lower: bigger batch absorbs this
    batch_size = {"4b": 8,    "12b": 4   }.get(model_size, 8)   # bumped up for 80GB VRAM

    ce = c_eps(eps)
    flip_rate = 0.0 if eps == float("inf") else 1 / (math.exp(eps) + 1)

    return {
        "learning_rate":                base_lr / ce,
        "gradient_accumulation_steps":  min(int(base_accum * ce ** 2), 32),
        "warmup_ratio":                 0.03 + 0.15 * flip_rate,
        "lr_scheduler_type":            "cosine",
        "max_grad_norm":                1.0,
        "num_train_epochs":             3,   # free under post-processing property
        "per_device_train_batch_size":  batch_size,
        "bf16":                         True,
        "gradient_checkpointing":       True,
    }
