"""Distilled shallow RSSM (4x smaller) for speculative drafting."""
import torch
import torch.nn as nn
import torch.distributions as td


class DraftRSSM(nn.Module):
    """Lightweight RSSM for fast speculative rollouts.
    
    4x smaller than target: smaller det_dim, stoch_dim, hidden_dim.
    Trained via distillation (KL matching to target).
    """
    def __init__(self, obs_dim, act_dim, det_dim=50, stoch_dim=16, hidden_dim=50):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.target_stoch_dim = None  # set during distillation
        
        # Minimal networks
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
        )
        self.posterior_net = nn.Linear(hidden_dim + det_dim, 2 * stoch_dim)
        self.prior_net = nn.Linear(det_dim, 2 * stoch_dim)
        self.gru = nn.GRUCell(stoch_dim + act_dim, det_dim)
        
        # Projection to align with target latent space
        self.project_to_target = nn.Linear(stoch_dim, 1)  # set dynamically
        self.project_from_target = nn.Linear(1, stoch_dim)
        self._projections_set = False
    
    def set_target_stoch_dim(self, target_stoch_dim):
        """Initialize projection layers to match target's stoch_dim."""
        self.target_stoch_dim = target_stoch_dim
        self.project_to_target = nn.Linear(self.stoch_dim, target_stoch_dim)
        self.project_from_target = nn.Linear(target_stoch_dim, self.stoch_dim)
        self._projections_set = True
    
    def initial_state(self, batch_size, device):
        det = torch.zeros(batch_size, self.det_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return det, stoch
    
    def imagine(self, prev_det, prev_stoch, action):
        """Single imagination step."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior_mean, prior_std = prior_params.chunk(2, dim=-1)
        prior_std = torch.softplus(prior_std) + 0.1
        prior = td.Independent(td.Normal(prior_mean, prior_std), 1)
        stoch = prior.rsample()
        return det, stoch, prior
    
    def imagine_deterministic(self, prev_det, prev_stoch, action):
        """Use mean instead of sample for more stable drafting."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior_mean, prior_std = prior_params.chunk(2, dim=-1)
        prior_std = torch.softplus(prior_std) + 0.1
        prior = td.Independent(td.Normal(prior_mean, prior_std), 1)
        stoch = prior_mean  # deterministic for drafting stability
        return det, stoch, prior
    
    def unroll_draft(self, det_0, stoch_0, actions):
        """Fast autoregressive drafting: actions [B, H, act_dim].
        
        Returns: priors list, (dets, stochs, target_stochs)
        """
        B, H, _ = actions.shape
        det, stoch = det_0, stoch_0
        priors = []
        dets, stochs, target_stochs = [], [], []
        
        for t in range(H):
            det, stoch, prior = self.imagine_deterministic(det, stoch, actions[:, t])
            priors.append(prior)
            dets.append(det)
            stochs.append(stoch)
            if self._projections_set:
                target_stochs.append(self.project_to_target(stoch))
        
        dets = torch.stack(dets, dim=1)
        stochs = torch.stack(stochs, dim=1)
        target_stochs = torch.stack(target_stochs, dim=1) if self._projections_set else None
        return priors, (dets, stochs, target_stochs)
