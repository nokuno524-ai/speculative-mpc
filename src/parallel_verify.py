"""Parallel verification of draft rollouts against target model."""
import torch
import torch.distributions as td


def parallel_verify_target(target_model, draft_target_stochs, actions, det_0, stoch_0):
    """Verify draft trajectory against target model.

    Re-runs target GRU sequentially using the draft's projected stochastic states
    as the stochastic input (simulating "what if we accepted the draft's states?").

    Args:
        target_model: The large target RSSM
        draft_target_stochs: [B, H, target_stoch_dim] — projected draft stoch states
        actions: [B, H, act_dim]
        det_0: [B, target_det_dim] — initial target deterministic state
        stoch_0: [B, target_stoch_dim] — initial target stochastic state

    Returns:
        target_dists: Independent Normal [B, H]
        target_dets: [B, H, target_det_dim]
    """
    B, H, _ = actions.shape
    det = det_0
    stoch = stoch_0  # initial stoch is in target space

    all_dets, all_means, all_stds = [], [], []

    for t in range(H):
        # Target GRU: feed draft's projected stoch as the stochastic state
        det = target_model.gru(
            torch.cat([stoch, actions[:, t]], dim=-1), det
        )
        # Target prior from this deterministic state
        prior = target_model.compute_prior_from_det(det)
        all_dets.append(det)
        all_means.append(prior.mean)
        all_stds.append(prior.stddev)

        # Use draft's projected stoch for next step (simulating acceptance)
        stoch = draft_target_stochs[:, t]

    target_dets = torch.stack(all_dets, dim=1)   # [B, H, det_dim]
    means = torch.stack(all_means, dim=1)         # [B, H, stoch_dim]
    stds = torch.stack(all_stds, dim=1)           # [B, H, stoch_dim]
    target_dists = td.Independent(td.Normal(means, stds), 1)
    return target_dists, target_dets
