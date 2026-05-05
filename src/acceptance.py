"""Adaptive KL threshold acceptance for speculative decoding."""
import torch
import torch.distributions as td


def evaluate_and_accept(target_dists, draft_dists, eps_base=0.1, alpha=0.01):
    """Evaluate draft vs target distributions with adaptive KL threshold.

    ε_t = ε₀ + α·t

    Args:
        target_dists: Independent Normal with batch_shape [B, H]
        draft_dists: Independent Normal with batch_shape [B, H]
    Returns:
        valid_prefix_mask: [B, H] bool
        accepted_lengths: [B] int
        kl_divs: [B, H] float
    """
    B, H = target_dists.batch_shape

    kl_divs = td.kl.kl_divergence(target_dists, draft_dists)  # [B, H]

    time_steps = torch.arange(H, device=kl_divs.device).float()
    adaptive_thresholds = eps_base + alpha * time_steps  # [H]
    adaptive_thresholds = adaptive_thresholds.unsqueeze(0).expand(B, H)

    accepted_mask = kl_divs < adaptive_thresholds
    valid_prefix_mask = torch.cumprod(accepted_mask.int(), dim=1).bool()
    accepted_lengths = valid_prefix_mask.sum(dim=1)

    return valid_prefix_mask, accepted_lengths, kl_divs
