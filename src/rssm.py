"""Minimal RSSM with GRU for continuous control."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class RSSM(nn.Module):
    """Recurrent State-Space Model with GRU transition.

    State = deterministic (GRU hidden) + stochastic (Gaussian).
    """

    def __init__(self, obs_dim, act_dim, det_dim=200, stoch_dim=30, hidden_dim=200):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Representation: obs -> stochastic posterior
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.posterior_net = nn.Linear(hidden_dim + det_dim, 2 * stoch_dim)

        # Transition prior: deterministic -> stochastic prior
        self.prior_net = nn.Linear(det_dim, 2 * stoch_dim)

        # GRU: (prev_stoch + action) -> deterministic state
        self.gru = nn.GRUCell(stoch_dim + act_dim, det_dim)

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size, device):
        det = torch.zeros(batch_size, self.det_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return det, stoch

    def observe(self, obs, prev_det, prev_stoch, action):
        """Representation step: obs + prev state + action -> posterior."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        obs_embed = self.obs_encoder(obs)
        post_params = self.posterior_net(torch.cat([obs_embed, det], dim=-1))
        posterior = self._parse_dist(post_params)
        stoch = posterior.rsample()
        return det, stoch, posterior

    def imagine(self, prev_det, prev_stoch, action):
        """Transition prior step: imagine next state without observation."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior = self._parse_dist(prior_params)
        stoch = prior.rsample()
        return det, stoch, prior

    def imagine_deterministic(self, prev_det, prev_stoch, action):
        """Like imagine but use mean (no sampling)."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior = self._parse_dist(prior_params)
        stoch = prior.mean
        return det, stoch, prior

    def get_reward(self, det, stoch):
        """Predict reward from state."""
        return self.reward_head(torch.cat([det, stoch], dim=-1)).squeeze(-1)

    def unroll_imagine(self, det_0, stoch_0, actions, deterministic=False):
        """Autoregressive imagination over H steps.

        Args:
            det_0, stoch_0: initial states [B, dim]
            actions: [B, H, act_dim]
        Returns:
            priors: list of H Independent Normal distributions
            dets: [B, H, det_dim]
            stochs: [B, H, stoch_dim]
        """
        B, H, _ = actions.shape
        fn = self.imagine_deterministic if deterministic else self.imagine
        det, stoch = det_0, stoch_0
        priors, dets, stochs = [], [], []
        for t in range(H):
            det, stoch, prior = fn(det, stoch, actions[:, t])
            priors.append(prior)
            dets.append(det)
            stochs.append(stoch)
        dets = torch.stack(dets, dim=1)
        stochs = torch.stack(stochs, dim=1)
        return priors, dets, stochs

    def compute_prior_from_det(self, det):
        """Compute prior distribution from deterministic state only."""
        prior_params = self.prior_net(det)
        return self._parse_dist(prior_params)

    def _parse_dist(self, params):
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return td.Independent(td.Normal(mean, std), 1)
