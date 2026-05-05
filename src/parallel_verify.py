"""Parallel verification of draft rollouts against target model."""
import torch
import torch.distributions as td


def parallel_verify_target(target_model, draft_dets, draft_stochs, actions, det_0, stoch_0):
    """Verify draft trajectory against target model in batched fashion.
    
    For each step t, compute target's transition prior given draft's (det_{t-1}, stoch_{t-1}, a_{t-1}).
    
    Since GRU is sequential, we re-run the target's GRU on the draft's stochastic states
    to get target-compatible deterministic states, then compute target priors.
    
    Args:
        target_model: The large target RSSM
        draft_dets: [B, H, det_dim_draft] — not used directly (wrong dim)
        draft_stochs: [B, H, target_stoch_dim] — projected draft stoch states  
        actions: [B, H, act_dim]
        det_0: [B, target_det_dim] — initial target deterministic state
        stoch_0: [B, target_stoch_dim] — initial target stochastic state
    
    Returns:
        target_dists: Independent Normal distributions [B, H]
        target_dets: [B, H, target_det_dim]
    """
    B, H, _ = actions.shape
    
    target_dets = []
    target_prior_means = []
    target_prior_stds = []
    
    det = det_0
    stoch = stoch_0
    
    for t in range(H):
        # Run target GRU with draft's stochastic state
        det = target_model.gru(
            torch.cat([stoch, actions[:, t]], dim=-1), det
        )
        # Compute target prior
        prior_params = target_model.prior_net(det)
        p_mean, p_std = prior_params.chunk(2, dim=-1)
        p_std = torch.softplus(p_std) + 0.1
        
        target_dets.append(det)
        target_prior_means.append(p_mean)
        target_prior_stds.append(p_std)
        
        # For next step, use the projected draft stochastic state
        # This simulates "what if we accepted the draft's state?"
        stoch = draft_stochs[:, t]
    
    target_dets = torch.stack(target_dets, dim=1)
    means = torch.stack(target_prior_means, dim=1)
    stds = torch.stack(target_prior_stds, dim=1)
    target_dists = td.Independent(td.Normal(means, stds), 1)
    
    return target_dists, target_dets
