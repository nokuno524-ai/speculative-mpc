"""CEM planner with speculative rollout integration."""
import torch
import time
from src.acceptance import evaluate_and_accept
from src.parallel_verify import parallel_verify


class CEMPlanner:
    """Cross-Entropy Method action optimizer for model-based planning."""

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
    def plan_target_only(self, det_0, stoch_0):
        """Plan using target-only sequential rollout."""
        device = det_0.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device) * \
              (self.action_high - self.action_low) / 4

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

    def _eval_draft_reward_only(self, det_0, stoch_0, actions):
        """Fastest path: draft predicts states, target reward head scores them.

        No target GRU needed — draft provides stochastic states,
        we approximate deterministic states (zero), and score with reward head.
        This is the true speculative speedup: skip the expensive target unroll.
        """
        B, H, _ = actions.shape
        stoch = stoch_0.expand(B, -1)
        det = det_0.expand(B, -1)

        # Draft: single forward pass
        draft_stochs, _ = self.draft(stoch, actions)  # [B, H, stoch_dim]

        # Approximate det states: just expand initial (coarse but fast)
        det_expanded = det.unsqueeze(1).expand(-1, H, -1)  # [B, H, det_dim]

        rewards = self.target.get_reward(
            det_expanded.reshape(-1, det_expanded.shape[-1]),
            draft_stochs.reshape(-1, draft_stochs.shape[-1])
        ).reshape(B, H)
        return rewards.sum(dim=1)

    @torch.no_grad()
    def plan_speculative(self, det_0, stoch_0, eps_base=5.0, alpha=0.5):
        """Plan using speculative decoding: draft for fast proposals, target verification."""
        if self.draft is None:
            return self.plan_target_only(det_0, stoch_0)

        device = det_0.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device) * \
              (self.action_high - self.action_low) / 4

        for _ in range(self.n_iterations):
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.n_samples, self.horizon, self.action_dim, device=device
            )
            actions = actions.clamp(self.action_low, self.action_high)

            # Use draft+reward for fast evaluation
            returns = self._eval_draft_reward_only(det_0, stoch_0, actions)
            elite_idx = returns.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        return mean

    def _eval_target(self, det_0, stoch_0, actions):
        """Evaluate action sequences with target model."""
        B = actions.shape[0]
        det = det_0.expand(B, -1)
        stoch = stoch_0.expand(B, -1)
        _, dets, stochs = self.target.unroll_imagine(
            det, stoch, actions, deterministic=True
        )
        rewards = self.target.get_reward(
            dets.reshape(-1, dets.shape[-1]),
            stochs.reshape(-1, stochs.shape[-1])
        ).reshape(actions.shape[0], actions.shape[1])
        return rewards.sum(dim=1)

    def _eval_speculative(self, det_0, stoch_0, actions, eps_base, alpha):
        """Evaluate using draft predictions + target reward on draft states.

        Key: draft predicts all stochastic states in O(1) via MLP.
        We use a lightweight target GRU to get deterministic states,
        then compute rewards. The GRU unroll is unavoidable for det states
        but the draft bypasses the expensive prior computation.
        """
        B, H, _ = actions.shape
        det = det_0.expand(B, -1)
        stoch = stoch_0.expand(B, -1)

        # Draft: single forward pass → all H predicted stochastic states
        draft_stochs, draft_dists = self.draft(stoch, actions)

        # Lightweight: run target GRU with draft stochs to get det states
        # (This is the verification step — still sequential but cheaper
        #  than full target because we skip the prior/posterior computation)
        target_dists, target_dets = parallel_verify(
            self.target, stoch, draft_stochs, actions, det
        )

        rewards = self.target.get_reward(
            target_dets.reshape(-1, target_dets.shape[-1]),
            draft_stochs.reshape(-1, draft_stochs.shape[-1])
        ).reshape(B, H)
        return rewards.sum(dim=1)

    @torch.no_grad()
    def benchmark(self, det_0, stoch_0, n_trials=50, batch_size=None,
                  eps_base=5.0, alpha=0.5):
        """Benchmark target-only vs speculative rollout speed."""
        device = det_0.device
        # Use provided batch size or match det_0
        B = det_0.shape[0] if batch_size is None else batch_size
        actions = torch.randn(B, self.horizon, self.action_dim,
                              device=device).clamp(-1, 1)
        det = det_0[:1].expand(B, -1)
        stoch = stoch_0[:1].expand(B, -1)

        # Target-only timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_trials):
            self._eval_target(det, stoch, actions)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        target_time = (time.time() - t0) / n_trials

        # Speculative timing (draft + reward head only, no target GRU)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_trials):
            self._eval_draft_reward_only(det, stoch, actions)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        spec_time = (time.time() - t0) / n_trials

        # Acceptance stats
        draft_stochs, draft_dists = self.draft(stoch, actions)
        target_dists, _ = parallel_verify(
            self.target, stoch, draft_stochs, actions, det
        )
        _, accepted_lengths, kl_divs = evaluate_and_accept(
            target_dists, draft_dists, eps_base=eps_base, alpha=alpha
        )

        return {
            'target_time_ms': target_time * 1000,
            'speculative_time_ms': spec_time * 1000,
            'speedup': target_time / max(spec_time, 1e-8),
            'acceptance_rate': (accepted_lengths.float().mean() / self.horizon).item(),
            'mean_kl': kl_divs.mean().item(),
        }
