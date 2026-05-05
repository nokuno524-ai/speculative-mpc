"""CEM planner with speculative rollout integration."""
import torch
import time


class CEMPlanner:
    """Cross-Entropy Method action optimizer."""

    def __init__(self, target_model, draft_model=None, horizon=12, n_samples=512,
                 n_iterations=5, elite_frac=0.1, action_dim=1,
                 action_low=-1.0, action_high=1.0):
        self.target = target_model
        self.draft = draft_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def plan(self, det_0, stoch_0):
        """Plan action sequence using target-only sequential rollout."""
        device = det_0.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device) * (self.action_high - self.action_low) / 4

        for _ in range(self.n_iterations):
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.n_samples, self.horizon, self.action_dim, device=device
            )
            actions = actions.clamp(self.action_low, self.action_high)

            returns = self._eval_target(det_0, stoch_0, actions)
            elite_idx = returns.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        return mean  # [H, action_dim]

    def _eval_target(self, det_0, stoch_0, actions):
        """Evaluate action sequences with target model."""
        B, H, _ = actions.shape
        priors, dets, stochs = self.target.unroll_imagine(
            det_0, stoch_0, actions, deterministic=True
        )
        rewards = []
        for t in range(H):
            rewards.append(self.target.get_reward(dets[:, t], stochs[:, t]))
        return torch.stack(rewards, dim=1).sum(dim=1)  # [B]

    @torch.no_grad()
    def benchmark(self, det_0, stoch_0, n_trials=50, batch_size=64):
        """Benchmark target-only vs speculative rollout speed."""
        from src.acceptance import speculative_rollout

        device = det_0.device
        actions = torch.randn(batch_size, self.horizon, self.action_dim, device=device).clamp(-1, 1)

        # Target-only timing
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        for _ in range(n_trials):
            self._eval_target(det_0, stoch_0, actions)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        target_time = (time.time() - t0) / n_trials

        # Speculative timing
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        for _ in range(n_trials):
            speculative_rollout(self.target, self.draft, det_0, stoch_0, actions)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        spec_time = (time.time() - t0) / n_trials

        # Acceptance stats (one run)
        _, _, stats = speculative_rollout(self.target, self.draft, det_0, stoch_0, actions)

        return {
            'target_time_ms': target_time * 1000,
            'speculative_time_ms': spec_time * 1000,
            'speedup': target_time / max(spec_time, 1e-8),
            'acceptance_rate': stats['avg_acceptance_rate'],
            'mean_kl': stats['mean_kl'],
        }
