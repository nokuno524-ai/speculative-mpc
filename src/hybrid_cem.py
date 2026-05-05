"""Round 4: Hybrid Coarse-to-Fine CEM Planner.

Uses jump-ahead model for coarse filtering, target model for fine re-evaluation.
"""
import torch
import time
import numpy as np


class HybridCEM:
    """CEM planner with coarse-to-fine evaluation.

    1. Generate N candidate action sequences
    2. Evaluate ALL with jump-ahead model (fast, coarse)
    3. Select top-K candidates
    4. Re-evaluate top-K with target model (slow, accurate)
    5. CEM update from target-evaluated elites
    """

    def __init__(self, target, jump_model, horizon=12, n_samples=512,
                 n_iterations=5, elite_frac=0.1, top_k=10,
                 action_dim=1, action_low=-1.0, action_high=1.0):
        self.target = target
        self.jump = jump_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.top_k = min(top_k, n_samples)
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def plan(self, det_0, stoch_0, track_time=False):
        """Plan with hybrid coarse-to-fine CEM."""
        device = det_0.device
        H = self.horizon
        N = self.n_samples

        mean = torch.zeros(H, self.action_dim, device=device)
        std = torch.ones(H, self.action_dim, device=device) * \
              (self.action_high - self.action_low) / 4

        timing = {'coarse_ms': 0, 'fine_ms': 0} if track_time else None

        for _ in range(self.n_iterations):
            # Sample actions
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                N, H, self.action_dim, device=device)
            actions = actions.clamp(self.action_low, self.action_high)

            # ── Coarse: jump-ahead evaluation of ALL candidates ──
            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()

            stoch_exp = stoch_0.expand(N, -1)
            from src.jump_ahead import jump_ahead_rollout
            _, coarse_rewards = jump_ahead_rollout(self.jump, stoch_exp, actions)
            coarse_returns = coarse_rewards.sum(dim=1)

            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                timing['coarse_ms'] += (time.time() - t0) * 1000

            # ── Select top-K ──
            topk_idx = coarse_returns.topk(min(self.top_k, N)).indices
            topk_actions = actions[topk_idx]  # [K, H, act_dim]

            # ── Fine: target model evaluation of top-K only ──
            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()

            K = topk_actions.shape[0]
            det_exp = det_0.expand(K, -1)
            stoch_exp_k = stoch_0.expand(K, -1)
            _, dets, stochs = self.target.unroll_imagine(
                det_exp, stoch_exp_k, topk_actions, deterministic=True)
            fine_rewards = self.target.get_reward(
                dets.reshape(-1, dets.shape[-1]),
                stochs.reshape(-1, stochs.shape[-1])
            ).reshape(K, H)
            fine_returns = fine_rewards.sum(dim=1)

            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                timing['fine_ms'] += (time.time() - t0) * 1000

            # CEM update from fine-evaluated elites
            n_elite = min(self.n_elite, K)
            elite_idx = fine_returns.topk(n_elite).indices
            elite_actions = topk_actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        return mean, timing


class TargetOnlyCEM:
    """Standard CEM with target model only (baseline)."""

    def __init__(self, target, horizon=12, n_samples=512,
                 n_iterations=5, elite_frac=0.1,
                 action_dim=1, action_low=-1.0, action_high=1.0):
        self.target = target
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def plan(self, det_0, stoch_0, track_time=False):
        device = det_0.device
        H = self.horizon
        N = self.n_samples
        timing = {'total_ms': 0} if track_time else None

        mean = torch.zeros(H, self.action_dim, device=device)
        std = torch.ones(H, self.action_dim, device=device) * \
              (self.action_high - self.action_low) / 4

        for _ in range(self.n_iterations):
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                N, H, self.action_dim, device=device)
            actions = actions.clamp(self.action_low, self.action_high)

            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()

            det_exp = det_0.expand(N, -1)
            stoch_exp = stoch_0.expand(N, -1)
            _, dets, stochs = self.target.unroll_imagine(
                det_exp, stoch_exp, actions, deterministic=True)
            rewards = self.target.get_reward(
                dets.reshape(-1, dets.shape[-1]),
                stochs.reshape(-1, stochs.shape[-1])
            ).reshape(N, H)
            returns = rewards.sum(dim=1)

            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                timing['total_ms'] += (time.time() - t0) * 1000

            elite_idx = returns.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        return mean, timing


class JumpOnlyCEM:
    """CEM with jump-ahead only (no target re-evaluation)."""

    def __init__(self, jump_model, horizon=12, n_samples=512,
                 n_iterations=5, elite_frac=0.1,
                 action_dim=1, action_low=-1.0, action_high=1.0):
        self.jump = jump_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_elite = max(1, int(n_samples * elite_frac))
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def plan(self, det_0, stoch_0, track_time=False):
        device = det_0.device
        H = self.horizon
        N = self.n_samples
        timing = {'total_ms': 0} if track_time else None

        mean = torch.zeros(H, self.action_dim, device=device)
        std = torch.ones(H, self.action_dim, device=device) * \
              (self.action_high - self.action_low) / 4

        from src.jump_ahead import jump_ahead_rollout

        for _ in range(self.n_iterations):
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                N, H, self.action_dim, device=device)
            actions = actions.clamp(self.action_low, self.action_high)

            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.time()

            stoch_exp = stoch_0.expand(N, -1)
            _, rewards = jump_ahead_rollout(self.jump, stoch_exp, actions)
            returns = rewards.sum(dim=1)

            if track_time:
                if device.type == 'cuda': torch.cuda.synchronize()
                timing['total_ms'] += (time.time() - t0) * 1000

            elite_idx = returns.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        return mean, timing
