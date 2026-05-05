"""Model loading utilities — builds correct architecture from checkpoint inspection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RSSMTarget(nn.Module):
    """RSSM that matches the trained checkpoint architecture.

    Auto-detected from checkpoint: deeper obs_encoder, posterior_net, prior_net, reward_head.
    """

    def __init__(self, obs_dim=4, act_dim=1, det_dim=200, stoch_dim=30, hidden_dim=256):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Deeper obs encoder matching checkpoint (6 layers: Linear+ELU x6, but last has no ELU)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Deeper posterior net matching checkpoint
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + det_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        # Deeper prior net matching checkpoint
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        self.gru = nn.GRUCell(stoch_dim + act_dim, det_dim)

        self.reward_head = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.det_dim, device=device),
                torch.zeros(batch_size, self.stoch_dim, device=device))

    def observe(self, obs, prev_det, prev_stoch, action):
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        obs_embed = self.obs_encoder(obs)
        post_params = self.posterior_net(torch.cat([obs_embed, det], dim=-1))
        posterior = self._parse_dist(post_params)
        stoch = posterior.rsample()
        return det, stoch, posterior

    def imagine(self, prev_det, prev_stoch, action):
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior = self._parse_dist(prior_params)
        stoch = prior.rsample()
        return det, stoch, prior

    def imagine_deterministic(self, prev_det, prev_stoch, action):
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior = self._parse_dist(prior_params)
        stoch = prior.mean
        return det, stoch, prior

    def get_reward(self, det, stoch):
        return self.reward_head(torch.cat([det, stoch], dim=-1)).squeeze(-1)

    def unroll_imagine(self, det_0, stoch_0, actions, deterministic=False):
        B, H, _ = actions.shape
        fn = self.imagine_deterministic if deterministic else self.imagine
        det, stoch = det_0, stoch_0
        priors, dets, stochs = [], [], []
        for t in range(H):
            det, stoch, prior = fn(det, stoch, actions[:, t])
            priors.append(prior)
            dets.append(det)
            stochs.append(stoch)
        return priors, torch.stack(dets, dim=1), torch.stack(stochs, dim=1)

    def compute_prior_from_det(self, det):
        return self._parse_dist(self.prior_net(det))

    def _parse_dist(self, params):
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return td.Independent(td.Normal(mean, std), 1)


def load_target(device='cpu'):
    model = RSSMTarget().to(device)
    path = os.path.join(PROJECT_ROOT, 'checkpoints', 'target_CartPole_.pt')
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_draft(device='cpu'):
    """Load the CausalConvDraft v2 (matches checkpoint)."""
    from src.ml_draft_v2 import CausalConvDraft
    model = CausalConvDraft(stoch_dim=30, act_dim=1, hidden_dim=128, n_layers=4, kernel_size=3).to(device)
    path = os.path.join(PROJECT_ROOT, 'checkpoints', 'draft_CartPole_.pt')
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


# Convenience
obs_dim, act_dim, det_dim, stoch_dim, hidden_dim = 4, 1, 200, 30, 256
