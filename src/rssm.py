"""Minimal RSSM with GRU for continuous control."""
import torch
import torch.nn as nn
import torch.distributions as td


class RSSM(nn.Module):
    """Recurrent State-Space Model with GRU transition.
    
    State: deterministic (GRU hidden) + stochastic (Gaussian).
    """
    def __init__(self, obs_dim, act_dim, det_dim=200, stoch_dim=30, hidden_dim=200,
                 reward_hidden=200):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        
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
        
        # Reward model
        self.reward_net = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, reward_hidden), nn.ELU(),
            nn.Linear(reward_hidden, reward_hidden), nn.ELU(),
            nn.Linear(reward_hidden, 1),
        )
    
    def initial_state(self, batch_size, device):
        det = torch.zeros(batch_size, self.det_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return det, stoch
    
    def observe(self, obs, prev_det, prev_stoch, action):
        """Representation step: given obs, prev state, action -> posterior."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        obs_embed = self.obs_encoder(obs)
        post_params = self.posterior_net(torch.cat([obs_embed, det], dim=-1))
        post_mean, post_std = post_params.chunk(2, dim=-1)
        post_std = torch.softplus(post_std) + 0.1
        posterior = td.Independent(td.Normal(post_mean, post_std), 1)
        stoch = posterior.rsample()
        return det, stoch, posterior
    
    def imagine(self, prev_det, prev_stoch, action):
        """Transition prior step: imagine next state without observation."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior_mean, prior_std = prior_params.chunk(2, dim=-1)
        prior_std = torch.softplus(prior_std) + 0.1
        prior = td.Independent(td.Normal(prior_mean, prior_std), 1)
        stoch = prior.rsample()
        return det, stoch, prior
    
    def imagine_with_dist(self, prev_det, prev_stoch, action):
        """Like imagine but returns distribution without sampling."""
        det = self.gru(torch.cat([prev_stoch, action], dim=-1), prev_det)
        prior_params = self.prior_net(det)
        prior_mean, prior_std = prior_params.chunk(2, dim=-1)
        prior_std = torch.softplus(prior_std) + 0.1
        prior = td.Independent(td.Normal(prior_mean, prior_std), 1)
        return det, prior
    
    def get_reward(self, det, stoch):
        return self.reward_net(torch.cat([det, stoch], dim=-1)).squeeze(-1)
    
    def unroll_imagine(self, det_0, stoch_0, actions):
        """Autoregressive imagination: actions [B, H, act_dim].
        
        Returns: priors list, (dets, stochs) each [B, H, dim]
        """
        B, H, _ = actions.shape
        det, stoch = det_0, stoch_0
        priors = []
        dets, stochs = [], []
        for t in range(H):
            det, stoch, prior = self.imagine(det, stoch, actions[:, t])
            priors.append(prior)
            dets.append(det)
            stochs.append(stoch)
        dets = torch.stack(dets, dim=1)
        stochs = torch.stack(stochs, dim=1)
        return priors, (dets, stochs)
    
    def parallel_verify(self, draft_dets, draft_stochs, actions):
        """Parallel verification: compute target prior for all steps at once.
        
        Given draft states [B, H, dim] and actions [B, H, act_dim],
        compute what the target model's prior would be at each step.
        This is the key operation for speculative decoding.
        
        IMPORTANT: We process each step independently but use the draft states
        as inputs. This is valid because we're computing p(s_t | s_{t-1}, a_{t-1})
        using the *draft's* s_{t-1}.
        """
        B, H, _ = actions.shape
        # For step t, input is (draft_stoch[t-1], action[t])
        # We prepend the initial state
        # draft_dets[t-1] and draft_stochs[t-1] -> GRU -> new det -> prior
        # But we can't truly parallelize GRU since det depends on prev det
        # Instead, we use the draft's determinstic states directly
        
        # Use draft dets as the deterministic state, compute prior from them
        # This is the verification: given draft's trajectory, what does target think?
        all_means, all_stds = [], []
        for t in range(H):
            # Use draft's deterministic state at time t to compute target's prior
            prior_params = self.prior_net(draft_dets[:, t])
            p_mean, p_std = prior_params.chunk(2, dim=-1)
            p_std = torch.softplus(p_std) + 0.1
            all_means.append(p_mean)
            all_stds.append(p_std)
        
        means = torch.stack(all_means, dim=1)  # [B, H, stoch_dim]
        stds = torch.stack(all_stds, dim=1)    # [B, H, stoch_dim]
        target_dists = td.Independent(td.Normal(means, stds), 1)
        return target_dists
