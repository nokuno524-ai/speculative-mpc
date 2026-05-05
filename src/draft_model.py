"""Distilled shallow RSSM for speculative drafting."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class DraftRSSM(nn.Module):
    """Lightweight RSSM (~4x smaller) trained via distillation.

    Has its own latent space with projections to/from target space.
    """

    def __init__(self, act_dim, det_dim=50, stoch_dim=16, target_stoch_dim=None):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.target_stoch_dim = target_stoch_dim

        self.prior_net = nn.Linear(det_dim, 2 * stoch_dim)
        self.gru = nn.GRUCell(stoch_dim + act_dim, det_dim)

        # Projections to align with target latent space
        if target_stoch_dim is not None:
            self.project_to_target = nn.Linear(stoch_dim, target_stoch_dim)
            self.project_from_target = nn.Linear(target_stoch_dim, stoch_dim)
        else:
            self.project_to_target = None
            self.project_from_target = None

    def initial_state(self, batch_size, device):
        det = torch.zeros(batch_size, self.det_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return det, stoch

    def imagine(self, prev_det, prev_stoch, action):
        """Single imagination step with sampling."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior = self._parse_dist(prior_params)
        stoch = prior.rsample()
        return det, stoch, prior

    def imagine_deterministic(self, prev_det, prev_stoch, action):
        """Single imagination step using mean (stable for drafting)."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior = self._parse_dist(prior_params)
        stoch = prior.mean  # deterministic for stable drafting
        return det, stoch, prior

    def unroll_draft(self, det_0, stoch_0, actions):
        """Fast autoregressive drafting with deterministic mean.

        Args:
            det_0, stoch_0: initial states [B, dim] (draft space)
            actions: [B, H, act_dim]

        Returns:
            target_priors: list of H Independent Normal distributions (in target stoch space)
            dets: [B, H, det_dim]
            stochs: [B, H, stoch_dim]  (draft space)
            target_stochs: [B, H, target_stoch_dim] or None
        """
        B, H, _ = actions.shape
        det, stoch = det_0, stoch_0
        target_priors, dets, stochs, target_stochs = [], [], [], []

        for t in range(H):
            det, stoch, prior = self.imagine_deterministic(det, stoch, actions[:, t])
            dets.append(det)
            stochs.append(stoch)
            if self.project_to_target is not None:
                tgt_s = self.project_to_target(stoch)
                target_stochs.append(tgt_s)
                # Project prior distribution to target space
                tgt_mean = self.project_to_target(prior.mean)
                # Approximate std via Jacobian (diagonal approximation)
                tgt_std = self.project_to_target(prior.stddev).abs() + 0.1
                target_priors.append(td.Independent(td.Normal(tgt_mean, tgt_std), 1))
            else:
                target_priors.append(prior)

        dets = torch.stack(dets, dim=1)
        stochs = torch.stack(stochs, dim=1)
        tgt = torch.stack(target_stochs, dim=1) if self.project_to_target is not None else None
        return target_priors, dets, stochs, tgt

    def _parse_dist(self, params):
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return td.Independent(td.Normal(mean, std), 1)
