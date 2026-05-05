"""Adaptive KL threshold acceptance for speculative decoding."""
import torch
import torch.distributions as td


def evaluate_and_accept(target_dists, draft_dists, eps_base=0.1, alpha=0.01):
    """Evaluate draft vs target distributions with adaptive KL threshold.
    
    ε_t = ε₀ + α·t  (threshold relaxes with horizon to account for compounding uncertainty)
    
    Args:
        target_dists: Target model distributions [B, H] (Independent Normal)
        draft_dists: Draft model distributions [B, H] (Independent Normal)
        eps_base: Base KL threshold (ε₀)
        alpha: Relaxation coefficient
    
    Returns:
        valid_prefix_mask: [B, H] bool — contiguous accepted prefix
        accepted_lengths: [B] int — number of accepted steps per batch
        kl_divs: [B, H] float — KL divergences
    """
    B, H = target_dists.batch_shape
    
    # KL(target || draft) — we want to know how far draft is from target
    kl_divs = td.kl.kl_divergence(target_dists, draft_dists)  # [B, H]
    
    # Adaptive threshold: relaxes over time
    time_steps = torch.arange(H, device=kl_divs.device).float()
    adaptive_thresholds = eps_base + alpha * time_steps  # [H]
    adaptive_thresholds = adaptive_thresholds.unsqueeze(0).expand(B, H)
    
    # Acceptance: KL must be below threshold
    accepted_mask = kl_divs < adaptive_thresholds  # [B, H]
    
    # Contiguous prefix: cumulative product ensures rejection at step k rejects all k' > k
    valid_prefix_mask = torch.cumprod(accepted_mask.int(), dim=1).bool()
    accepted_lengths = valid_prefix_mask.sum(dim=1)  # [B]
    
    return valid_prefix_mask, accepted_lengths, kl_divs


def speculative_rollout(target_model, draft_model, det_0, stoch_0, actions,
                        eps_base=0.1, alpha=0.01):
    """Full speculative rollout: draft -> verify -> accept/resample.
    
    Args:
        target_model: Large target RSSM
        draft_model: Small draft RSSM
        det_0: Initial deterministic state (target dim)
        stoch_0: Initial stochastic state (target dim)
        actions: [B, H, act_dim] proposed actions
        eps_base, alpha: Adaptive threshold params
    
    Returns:
        final_dets: [B, H, det_dim] — accepted deterministic states (target)
        final_stochs: [B, H, stoch_dim] — accepted stochastic states (target)
        stats: dict with acceptance rates, KL values, speedup estimate
    """
    from src.parallel_verify import parallel_verify_target
    
    B, H, _ = actions.shape
    device = actions.device
    
    # Project initial state to draft space
    draft_stoch_0 = draft_model.project_from_target(stoch_0)
    draft_det_0 = torch.zeros(B, draft_model.det_dim, device=device)
    
    # Step 1: Draft full trajectory
    draft_priors, (draft_dets, draft_stochs, draft_target_stochs) = draft_model.unroll_draft(
        draft_det_0, draft_stoch_0, actions
    )
    
    # Step 2: Verify in parallel against target
    target_dists, target_dets = parallel_verify_target(
        target_model, draft_dets, draft_target_stochs, actions, det_0, stoch_0
    )
    
    # Create draft dists tensor
    draft_means = torch.stack([p.mean for p in draft_priors], dim=1)
    draft_stds_tensor = torch.stack([p.stddev for p in draft_priors], dim=1)
    draft_dists = td.Independent(td.Normal(draft_means, draft_stds_tensor), 1)
    
    # Step 3: Accept/reject
    valid_mask, accepted_lengths, kl_divs = evaluate_and_accept(
        target_dists, draft_dists, eps_base=eps_base, alpha=alpha
    )
    
    # Step 4: Build final trajectory — use draft where accepted, target where rejected
    # For simplicity in v1: use target's stochastic states everywhere
    # (since we already computed target priors)
    # Resample from target where accepted, use target mean where rejected
    final_stochs = draft_target_stochs.clone()
    
    # For rejected steps beyond the prefix, resample from target
    for b in range(B):
        k = accepted_lengths[b].item()
        if k < H:
            # Use target distribution to sample remaining states
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
