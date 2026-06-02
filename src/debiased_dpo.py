"""
De-biased Square-Loss DPO Trainer (SquareχPO-style).

Implements Algorithm 1 from SquareχPO (arXiv:2505.21395) for the local DP (CTL/LTC) setting.
One-line change from standard DPO: log-loss replaced by de-biased square-loss over probability.

Loss: [2σ(clip_{2R}[β·h]) - 1 - c(ε)·z_i]²
where c(ε) = (e^ε+1)/(e^ε-1), z_i ∈ {-1,+1} are noisy labels post RR.
"""

import math
import torch
from trl import DPOTrainer
from typing import Optional


def c_eps(eps: float) -> float:
    """De-biasing factor c(ε) = (e^ε+1)/(e^ε-1). Returns 1.0 for ε=∞."""
    if eps == float("inf") or eps > 100:
        return 1.0
    return (math.exp(eps) + 1) / (math.exp(eps) - 1)


class DebiasedSquareDPOTrainer(DPOTrainer):
    """
    SquareχPO de-biased square-loss DPO trainer.

    Args:
        eps: Privacy budget ε. Use float('inf') for no-privacy (standard DPO equivalent).
        R_max: Reward clipping bound. Default 1.0.
        All other args passed to DPOTrainer.
    """

    def __init__(self, *args, eps: float = float("inf"), R_max: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.R_max = R_max
        self._c_eps = c_eps(eps)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        reference_free: bool = False,
    ):
        """
        Compute de-biased square-loss DPO loss.

        Expects batch to contain 'noisy_label' field (±1 floats, pre-computed by RR).
        Falls back to standard chosen=+1 if not present (for ε=∞ clean baseline).
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if not reference_free:
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            h = pi_logratios - ref_logratios
        else:
            h = pi_logratios

        # Retrieve noisy labels from the stored batch, or use clean +1
        if hasattr(self, "_current_noisy_labels") and self._current_noisy_labels is not None:
            z = self._current_noisy_labels.to(h.device).float()
        else:
            z = torch.ones_like(h)  # clean labels: all chosen = +1

        h_scaled = self.beta * h
        h_clipped = torch.clamp(h_scaled, -2 * self.R_max, 2 * self.R_max)

        # De-biased square-loss: [2σ(h_clipped) - 1 - c(ε)·z]²
        loss = (2 * torch.sigmoid(h_clipped) - 1 - self._c_eps * z) ** 2

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Stash noisy labels for use inside dpo_loss
        self._current_noisy_labels = inputs.pop("noisy_label", None)
        result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                      num_items_in_batch=num_items_in_batch)
        self._current_noisy_labels = None
        return result
