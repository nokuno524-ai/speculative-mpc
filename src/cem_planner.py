"""Cross-Entropy Method planner with speculative rollout integration."""
import torch
import torch.distributions as td
from src.acceptance import speculative_rollout


class CEMPlanner:
    """CEM action optimizer using speculative decoding for fast rollouts.
    
    Compares:
    1. Target-only: Sequential rollout with target RSSM (slow baseline)
    2. Speculative: Draft proposes, target verifies in parallel (fast)
    """
    def __init__(self, target_model, draft_model, horizon=12, n_samples=512,
                 n_iterations=5, elite_frac=0.1, eps_base=0.1, alpha=0.01,
                 action_dim=1, action_low=-1.0, action_high=1.0):
        self.target = target_model
        self.draft = draft_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.elite_frac = elite_frac
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.eps_base = eps_base
        self.alpha = alpha
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
    
    def _evaluate_actions_speculative(self, det_0, stoch_0, actions):
        """Evaluate action sequences using speculative decoding."""
        _, _, stats = speculative_rollout(
            self.target, self.draft, det_0, stoch_0, actions,
            eps_base=self.eps_base, alpha=self.alpha
        )
        return stats
    
    def _evaluate_actions_target_only(self, det_0, stoch_0, actions):
        """Sequential evaluation with target only (baseline)."""
        B, H, _ = actions.shape
        det, stoch = det_0, stoch_0
        rewards = []
        for t in range(H):
            det, stoch, _ = self.target.imagine(det, stoch, actions[:, t])
            reward = self.target.get_reward(det, stoch)
            rewards.append(reward)
        returns = torch.stack(rewards, dim=1).sum(dim=1)
        return returns
    
    @torch.no_grad()
    def plan_speculative(self, det_0, stoch_0):
        """Plan using speculative decoding."""
        device = det_0.device
        
        # Initialize action distribution
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device) * (self.action_high - self.action_low) / 4
        
        for iteration in range(self.n_iterations):
            # Sample actions
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.n_samples, self.horizon, self.action_dim, device=device
            )
            actions = actions.clamp(self.action_low, self.action_high)
            
            # Evaluate with target only (for correct reward)
            returns = self._evaluate_actions_target_only(det_0, stoch_0, actions)
            
            # Select elites
            elite_idx = returns.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]
            
            # Update distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6
        
        return mean  # [H, action_dim] — best action sequence
    
    @torch.no_grad()
    def plan_with_speculative_benchmark(self, det_0, stoch_0):
        """Plan and benchmark speculative vs target-only."""
        import time
        
        device = det_0.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device) * (self.action_high - self.action_low) / 4
        
        # Benchmark target-only rollout
        test_actions = mean.unsqueeze(0).expand(64, -1, -1) + std.unsqueeze(0) * torch.randn(
            64, self.horizon, self.action_dim, device=device
        )
        
        t0 = time.time()
        for _ in range(10):
            self._evaluate_actions_target_only(det_0, stoch_0, test_actions)
        target_time = (time.time() - t0) / 10
        
        t0 = time.time()
        for _ in range(10):
            self._evaluate_actions_speculative(det_0, stoch_0, test_actions)
        speculative_time = (time.time() - t0) / 10
        
        speedup = target_time / max(speculative_time, 1e-8)
        
        return {
            'target_time_ms': target_time * 1000,
            'speculative_time_ms': speculative_time * 1000,
            'speedup': speedup,
        }
