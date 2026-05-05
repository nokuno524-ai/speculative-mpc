"""Adaptive KL threshold acceptance for speculative decoding."""
import torch
import torch.distributions as td


def evaluate_and_accept(target_dists, draft_dists, eps_base=0.1, alpha=0.01):
    """Evaluate draft vs target distributions with adaptive KL threshold.

    ε_t = ε₀ + α·t

    Args:
        target_dists: [B, H] Independent Normal
        draft_dists: [B, H] Independent Normal
        eps_base: Base KL threshold
        alpha: Relaxation coefficient

    Returns:
        valid_prefix_mask: [B, H] bool
        accepted_lengths: [B] int
        kl_divs: [B, H] float
    """
    B, H = target_dists.batch_shape

    kl_divs = td.kl.kl_divergence(target_dists, draft_dists)  # [B, H]

    # Adaptive threshold: relaxes over time
    time_steps = torch.arange(H, device=kl_divs.device).float()
    adaptive_thresholds = eps_base + alpha * time_steps  # [H]
    adaptive_thresholds = adaptive_thresholds.unsqueeze(0).expand(B, H)

    accepted_mask = kl_divs < adaptive_thresholds
    valid_prefix_mask = torch.cumprod(accepted_mask.int(), dim=1).bool()
    accepted_lengths = valid_prefix_mask.sum(dim=1)

    return valid_prefix_mask, accepted_lengths, kl_divs


def make_draft_dists_tensor(draft_priors):
    """Convert list of H priors to a single [B, H] Independent Normal."""
    means = torch.stack([p.mean for p in draft_priors], dim=1)
    stds = torch.stack([p.stddev for p in draft_priors], dim=1)
    return td.Independent(td.Normal(means, stds), 1)


def speculative_rollout(target_model, draft_model, det_0, stoch_0, actions,
                        eps_base=0.1, alpha=0.01):
    """Full speculative rollout: draft → verify → accept/resample.

    Args:
        target_model: Large target RSSM
        draft_model: Small draft RSSM (must have project_to_target set)
        det_0: [B, target_det_dim] initial target deterministic state
        stoch_0: [B, target_stoch_dim] initial target stochastic state
        actions: [B, H, act_dim]
        eps_base, alpha: Adaptive threshold params

    Returns:
        final_dets: [B, H, target_det_dim]
        final_stochs: [B, H, target_stoch_dim]
        stats: dict
    """
    from src.parallel_verify import parallel_verify_target

    B, H, _ = actions.shape
    device = actions.device

    # Project initial stoch to draft space
    draft_stoch_0 = draft_model.project_from_target(stoch_0)
    draft_det_0 = torch.zeros(B, draft_model.det_dim, device=device)

    # Step 1: Draft full trajectory
    draft_priors, draft_dets, draft_stochs, draft_target_stochs = draft_model.unroll_draft(
        draft_det_0, draft_stoch_0, actions
    )

    # Step 2: Verify against target
    target_dists, target_dets = parallel_verify_target(
        target_model, draft_target_stochs, actions, det_0, stoch_0
    )

    # Step 3: Build draft dists tensor and accept/reject
    draft_dists = make_draft_dists_tensor(draft_priors)
    valid_mask, accepted_lengths, kl_divs = evaluate_and_accept(
        target_dists, draft_dists, eps_base=eps_base, alpha=alpha
    )

    # Step 4: Build final trajectory — draft where accepted, target mean where rejected
    final_stochs = draft_target_stochs.clone()
    for b in range(B):
        k = accepted_lengths[b].item()
        if k < H:
            final_stochs[b, k:] = target_dists.mean[b, k:]

    avg_acceptance = accepted_lengths.float().mean() / H
    stats = {
        'avg_acceptance_rate': avg_acceptance.item(),
        'mean_kl': kl_divs.mean().item(),
        'max_kl': kl_divs.max().item(),
        'accepted_lengths': accepted_lengths,
        'kl_divs': kl_divs,
        'speculative_steps': accepted_lengths.float().mean().item(),
        'total_steps': H,
    }

    return target_dets, final_stochs, stats
