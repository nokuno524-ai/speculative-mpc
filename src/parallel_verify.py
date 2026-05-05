"""Parallel verification of draft rollouts against target model.

Runs target GRU sequentially but uses draft's stochastic states as input,
producing target distributions for KL comparison.
"""
import torch
import torch.distributions as td


def parallel_verify(target_model, stoch_0, draft_stochs, actions, det_0):
    """Verify draft trajectory against target model.

    Runs target GRU sequentially, feeding draft's predicted stochastic states
    as the stochastic input at each step. Produces target prior distributions.

    Args:
        target_model: The target RSSM
        stoch_0: [B, stoch_dim] initial target stochastic state
        draft_stochs: [B, H, stoch_dim] predicted draft stochastic states
        actions: [B, H, act_dim]
        det_0: [B, det_dim] initial target deterministic state

    Returns:
        target_dists: Independent Normal with batch_shape [B, H]
        target_dets: [B, H, det_dim]
    """
    B, H, _ = actions.shape
    det = det_0
    stoch = stoch_0

    all_dets, all_means, all_stds = [], [], []

    for t in range(H):
        det = target_model.gru(
            torch.cat([stoch, actions[:, t]], dim=-1), det
        )
        prior = target_model.compute_prior_from_det(det)
        all_dets.append(det)
        all_means.append(prior.mean)
        all_stds.append(prior.stddev)
        # Feed draft's predicted stoch for next step
        stoch = draft_stochs[:, t]

    target_dets = torch.stack(all_dets, dim=1)
    means = torch.stack(all_means, dim=1)
    stds = torch.stack(all_stds, dim=1)
    target_dists = td.Independent(td.Normal(means, stds), 1)
    return target_dists, target_dets
