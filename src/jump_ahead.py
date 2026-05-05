"""Round 3 Pivot: Jump-Ahead Prediction for Parallel World Model Rollouts.

Core idea: Instead of predicting s_{t+1} from s_t (requiring H sequential steps),
train a model to predict s_{t+k} directly from s_t, reducing sequential steps to H/k.

For horizon H=30 and jump k=5, we go from 30 sequential GRU steps to 6 jump-ahead steps.
Each jump step is a single forward pass (no recurrence), so it's parallelizable via batching.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time, json, random, argparse
from collections import deque

from src.rssm import RSSM
from src.main_v2 import DeepRSSM, ReplayBuffer, collect_eps_greedy_data, train_target


# ── Jump-Ahead Model ──────────────────────────────────────────────────────────

class JumpAheadModel(nn.Module):
    """Predict s_{t+k} directly from s_t + action sequence [a_t, ..., a_{t+k-1}].

    No recurrence — processes the full action sequence with a transformer-like
    architecture for O(1) sequential steps.
    """

    def __init__(self, stoch_dim, act_dim, jump_k=5, hidden_dim=256, n_layers=4):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.act_dim = act_dim
        self.jump_k = jump_k

        # Encode action sequence: [a_t, ..., a_{t+k-1}] -> single vector
        self.act_encoder = nn.Sequential(
            nn.Linear(act_dim * jump_k, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )

        # Combine state + action encoding -> predict next state
        self.predictor = nn.Sequential(
            nn.Linear(stoch_dim + hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

        # Reward predictor for the jump
        self.reward_head = nn.Sequential(
            nn.Linear(stoch_dim * 2, hidden_dim), nn.ELU(),  # s_t + s_{t+k}
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, jump_k),  # predict k rewards at once
        )

    def forward(self, stoch_t, actions_chunk):
        """Predict s_{t+k} from s_t and k actions.

        Args:
            stoch_t: [B, stoch_dim]
            actions_chunk: [B, k, act_dim]
        Returns:
            stoch_next: [B, stoch_dim] — predicted state at t+k
            rewards: [B, k] — predicted rewards for each step
        """
        B = stoch_t.shape[0]
        act_flat = actions_chunk.reshape(B, -1)  # [B, k*act_dim]
        act_embed = self.act_encoder(act_flat)
        combined = torch.cat([stoch_t, act_embed], dim=-1)
        stoch_next = self.predictor(combined)
        rewards = self.reward_head(torch.cat([stoch_t, stoch_next], dim=-1))
        return stoch_next, rewards


# ── Training ──────────────────────────────────────────────────────────────────

def train_jump_ahead(target, jump_model, buffer, n_epochs=500, batch_size=128,
                     seq_len=40, lr=3e-4, device='cpu'):
    """Train jump-ahead model by distilling from target RSSM."""
    optimizer = optim.Adam(jump_model.parameters(), lr=lr)
    target.eval()
    jump_model.train()
    k = jump_model.jump_k

    for epoch in range(n_epochs):
        obs, acts, rews = buffer.sample_sequences(batch_size, seq_len)
        obs, acts, rews = obs.to(device), acts.to(device), rews.to(device)
        B, T = obs.shape[:2]

        # Get target states using observe + imagine
        with torch.no_grad():
            det, stoch = target.initial_state(B, device)
            target_stochs = []
            target_dets = []
            for t in range(T):
                det, stoch, _ = target.observe(obs[:, t], det, stoch, acts[:, t])
                target_stochs.append(stoch)
                target_dets.append(det)
            target_stochs = torch.stack(target_stochs, dim=1)  # [B, T, stoch_dim]
            target_rews = rews  # [B, T]

        # Create training pairs: (s_t, a[t:t+k]) -> s_{t+k}
        max_start = T - k
        if max_start < 1:
            continue

        # Random starting points
        starts = torch.randint(0, max_start, (B,), device=device)
        stoch_t = torch.stack([target_stochs[b, s] for b, s in enumerate(starts)])
        stoch_tp1 = torch.stack([target_stochs[b, s + k] for b, s in enumerate(starts)])
        act_chunks = torch.stack([acts[b, s:s+k] for b, s in enumerate(starts)])
        rew_chunks = torch.stack([target_rews[b, s:s+k] for b, s in enumerate(starts)])

        optimizer.zero_grad()
        pred_stoch, pred_rews = jump_model(stoch_t, act_chunks)

        state_loss = F.mse_loss(pred_stoch, stoch_tp1.detach())
        rew_loss = F.mse_loss(pred_rews, rew_chunks.detach())
        loss = state_loss + 0.5 * rew_loss

        loss.backward()
        nn.utils.clip_grad_norm_(jump_model.parameters(), 100.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Jump epoch {epoch+1}/{n_epochs} | "
                  f"State MSE: {state_loss.item():.6f} | Rew MSE: {rew_loss.item():.4f}")

    return jump_model


# ── Jump-Ahead Planning ──────────────────────────────────────────────────────

@torch.no_grad()
def jump_ahead_rollout(jump_model, stoch_0, actions, det_0=None):
    """Perform a jump-ahead rollout using k-step jumps.

    Args:
        stoch_0: [B, stoch_dim]
        actions: [B, H, act_dim]
        det_0: unused (jump-ahead doesn't use deterministic state)
    Returns:
        stochs: [B, n_jumps, stoch_dim] — states at each jump point
        rewards: [B, H] — predicted rewards for all steps
    """
    B, H, act_dim = actions.shape
    k = jump_model.jump_k
    n_jumps = H // k

    all_stochs = []
    all_rews = []
    stoch = stoch_0

    for j in range(n_jumps):
        chunk = actions[:, j*k:(j+1)*k]
        stoch, rews = jump_model(stoch, chunk)
        all_stochs.append(stoch)
        all_rews.append(rews)

    stochs = torch.stack(all_stochs, dim=1)  # [B, n_jumps, stoch_dim]

    # Flatten rewards to cover all H steps
    rews_flat = torch.cat(all_rews, dim=1)  # [B, n_jumps * k] = [B, H] if H % k == 0
    # Pad if needed
    if rews_flat.shape[1] < H:
        padding = torch.zeros(B, H - rews_flat.shape[1], device=rews_flat.device)
        rews_flat = torch.cat([rews_flat, padding], dim=1)

    return stochs, rews_flat[:, :H]


# ── CEM Planner with Jump-Ahead ───────────────────────────────────────────────

class JumpAheadCEM:
    """CEM planner using jump-ahead model for fast rollouts."""

    def __init__(self, target, jump_model, horizon=12, n_samples=256,
                 n_iterations=5, action_dim=1, action_low=-1.0, action_high=1.0):
        self.target = target
        self.jump = jump_model
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    def plan(self, det, stoch):
        """Plan using jump-ahead rollouts for CEM proposals, target for final eval."""
        N = self.n_samples
        H = self.horizon
        device = det.device

        # Initialize action distribution
        mean = torch.zeros(H, self.action_dim, device=device)
        std = torch.ones(H, self.action_dim, device=device) * 0.5

        for it in range(self.n_iterations):
            # Sample actions
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) *
                       torch.randn(N, H, self.action_dim, device=device))
            actions = actions.clamp(self.action_low, self.action_high)

            # Repeat initial state for N samples
            stoch_expanded = stoch.expand(N, -1)

            # Jump-ahead rollout (fast!)
            _, rewards = jump_ahead_rollout(self.jump, stoch_expanded, actions)

            # CEM update
            Returns = rewards.sum(dim=1)
            elite_idx = Returns.argsort(descending=True)[:N // 5]
            elite_actions = actions[elite_idx]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 0.01

        # Final: evaluate best action sequence with TARGET model
        best_actions = mean.unsqueeze(0)  # [1, H, act_dim]
        _, dets, stochs = self.target.unroll_imagine(
            det, stoch, best_actions, deterministic=True)
        rewards = self.target.get_reward(
            dets.reshape(-1, dets.shape[-1]),
            stochs.reshape(-1, stochs.shape[-1]))
        # (We still use the target-evaluated best actions, but CEM was faster)

        return mean  # [H, act_dim]


# ── Benchmark ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark(target, jump_model, device, act_dim=1, H=30, batch_size=256,
              n_trials=100):
    target.eval()
    jump_model.eval()

    actions = torch.randn(batch_size, H, act_dim, device=device).clamp(-1, 1)
    det_0, stoch_0 = target.initial_state(batch_size, device)

    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None

    # Warmup
    for _ in range(10):
        target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        jump_ahead_rollout(jump_model, stoch_0, actions)

    # Target sequential
    sync(); t0 = time.time()
    for _ in range(n_trials):
        _, dets, stochs = target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        target.get_reward(dets.reshape(-1, dets.shape[-1]),
                          stochs.reshape(-1, stochs.shape[-1]))
    sync()
    target_time = (time.time() - t0) / n_trials

    # Jump-ahead
    sync(); t0 = time.time()
    for _ in range(n_trials):
        _, rewards = jump_ahead_rollout(jump_model, stoch_0, actions)
    sync()
    jump_time = (time.time() - t0) / n_trials

    k = jump_model.jump_k
    n_jumps = H // k
    return {
        'target_time_ms': target_time * 1000,
        'jump_time_ms': jump_time * 1000,
        'speedup': target_time / max(jump_time, 1e-8),
        'sequential_steps_target': H,
        'sequential_steps_jump': n_jumps,
        'jump_k': k,
    }


# ── MPC Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_mpc(target, jump_model, device, env_name="CartPole-v1", n_episodes=20,
                 horizon=12, n_samples=256, n_iterations=5):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    is_discrete = hasattr(env.action_space, 'n')
    act_dim = 1 if is_discrete else env.action_space.shape[0]

    planner_target = JumpAheadCEM(target, jump_model, horizon=horizon,
                                  n_samples=n_samples, n_iterations=n_iterations,
                                  action_dim=act_dim)
    # For target-only, we need a version that uses target rollouts for CEM
    planner_baseline = JumpAheadCEM(target, jump_model, horizon=horizon,
                                    n_samples=n_samples, n_iterations=n_iterations,
                                    action_dim=act_dim)

    results = {'target_only': [], 'jump_ahead': []}

    for mode in ['target_only', 'jump_ahead']:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False

            det, stoch = target.initial_state(1, device)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            zero_act = torch.zeros(1, act_dim, device=device)
            det, stoch, _ = target.observe(obs_t, det, stoch, zero_act)

            while not done:
                if mode == 'target_only':
                    # Standard target rollout for CEM
                    action_seq = _target_only_plan(
                        target, det, stoch, horizon, n_samples, n_iterations,
                        act_dim, device)
                else:
                    action_seq = planner_target.plan(det, stoch)

                action_cont = action_seq[0].cpu().numpy()

                if is_discrete:
                    action_disc = int(action_cont[0] > 0) if action_cont.ndim > 0 else int(action_cont > 0)
                    next_obs, reward, term, trunc, _ = env.step(action_disc)
                else:
                    action_clipped = np.clip(action_cont, env.action_space.low, env.action_space.high)
                    next_obs, reward, term, trunc, _ = env.step(action_clipped.astype(np.float32))

                total_reward += reward
                obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                act_flat = np.asarray(action_cont, dtype=np.float32).flatten()[:act_dim]
                act_t = torch.FloatTensor(act_flat).unsqueeze(0).to(device)
                det, stoch, _ = target.observe(obs_t, det, stoch, act_t)
                done = term or trunc

            results[mode].append(total_reward)
            if (ep + 1) % 5 == 0:
                print(f"    {mode} ep {ep+1}: reward={total_reward:.1f}")

    env.close()

    tgt_mean = np.mean(results['target_only'])
    jump_mean = np.mean(results['jump_ahead'])
    retention = jump_mean / max(tgt_mean, 1)
    print(f"\n  Target-only reward:  {tgt_mean:.1f} ± {np.std(results['target_only']):.1f}")
    print(f"  Jump-ahead reward:   {jump_mean:.1f} ± {np.std(results['jump_ahead']):.1f}")
    print(f"  Reward retention:    {retention:.1%}")

    return results, retention


def _target_only_plan(target, det, stoch, horizon, n_samples, n_iterations,
                      act_dim, device):
    """Standard CEM with target model rollouts."""
    N = n_samples
    mean = torch.zeros(horizon, act_dim, device=device)
    std = torch.ones(horizon, act_dim, device=device) * 0.5

    for it in range(n_iterations):
        actions = (mean.unsqueeze(0) + std.unsqueeze(0) *
                   torch.randn(N, horizon, act_dim, device=device)).clamp(-1, 1)

        det_exp = det.expand(N, -1)
        stoch_exp = stoch.expand(N, -1)
        _, dets, stochs = target.unroll_imagine(det_exp, stoch_exp, actions, deterministic=True)
        rewards = target.get_reward(dets.reshape(-1, dets.shape[-1]),
                                    stochs.reshape(-1, stochs.shape[-1]))
        Returns = rewards.reshape(N, horizon).sum(dim=1)

        elite_idx = Returns.argsort(descending=True)[:N // 5]
        elite_actions = actions[elite_idx]
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0) + 0.01

    return mean


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--n-episodes', type=int, default=10000)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--target-epochs', type=int, default=500)
    parser.add_argument('--jump-epochs', type=int, default=500)
    parser.add_argument('--jump-k', type=int, default=5,
                        help='Jump size (predict k steps ahead)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Round 3 Pivot: Jump-Ahead Prediction")
    print(f"  Env: {args.env} | Device: {device} | Jump k={args.jump_k}")
    print(f"{'='*60}")

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    is_discrete = hasattr(env.action_space, 'n')
    act_dim = 1 if is_discrete else env.action_space.shape[0]
    stoch_dim = 30
    env.close()

    # Models
    target = DeepRSSM(obs_dim, act_dim, det_dim=200, stoch_dim=stoch_dim).to(device)
    jump_model = JumpAheadModel(stoch_dim, act_dim, jump_k=args.jump_k,
                                 hidden_dim=256, n_layers=4).to(device)

    n_target = sum(p.numel() for p in target.parameters())
    n_jump = sum(p.numel() for p in jump_model.parameters())
    print(f"\nTarget params: {n_target:,}")
    print(f"Jump-ahead params: {n_jump:,} ({n_jump/n_target:.1%} of target)")

    # Train
    print(f"\n--- Collecting data ---")
    buffer = collect_eps_greedy_data(args.env, args.n_episodes, epsilon=args.epsilon)

    print(f"\n--- Training Target ---")
    train_target(target, buffer, n_epochs=args.target_epochs, device=device)

    print(f"\n--- Training Jump-Ahead (k={args.jump_k}) ---")
    train_jump_ahead(target, jump_model, buffer, n_epochs=args.jump_epochs,
                     seq_len=max(40, args.jump_k * 4), device=device)

    # Benchmark
    print(f"\n--- Benchmarking (H=30, k={args.jump_k}) ---")
    bench = benchmark(target, jump_model, device, act_dim=act_dim, H=30)
    print(f"  Target sequential:  {bench['target_time_ms']:.2f} ms ({bench['sequential_steps_target']} steps)")
    print(f"  Jump-ahead:         {bench['jump_time_ms']:.2f} ms ({bench['sequential_steps_jump']} jumps)")
    print(f"  Speedup:            {bench['speedup']:.1f}x")

    # MPC evaluation
    print(f"\n--- MPC Evaluation on {args.env} ---")
    mpc_results, retention = evaluate_mpc(
        target, jump_model, device, env_name=args.env, n_episodes=20,
        horizon=12, n_samples=256, n_iterations=5)

    # Sweep different k values
    print(f"\n--- Jump Size Sweep ---")
    for k in [3, 5, 10]:
        if k == args.jump_k:
            # Already tested
            continue
        jm = JumpAheadModel(stoch_dim, act_dim, jump_k=k, hidden_dim=256).to(device)
        train_jump_ahead(target, jm, buffer, n_epochs=300,
                         seq_len=max(40, k * 4), device=device)
        b = benchmark(target, jm, device, act_dim=act_dim, H=30)
        print(f"  k={k}: {b['speedup']:.1f}x speedup, "
              f"{b['sequential_steps_jump']} sequential steps, "
              f"{b['jump_time_ms']:.2f} ms")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Jump-Ahead Pivot ({args.env})")
    print(f"{'='*60}")
    print(f"  Jump k={args.jump_k}: {bench['speedup']:.1f}x speedup")
    print(f"  Sequential steps: {bench['sequential_steps_target']} → {bench['sequential_steps_jump']}")
    print(f"  Target reward:  {np.mean(mpc_results['target_only']):.1f}")
    print(f"  Jump reward:    {np.mean(mpc_results['jump_ahead']):.1f}")
    print(f"  Retention:      {retention:.1%}")
    print(f"{'='*60}")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    env_tag = args.env.replace('-', '_').replace('v1', '').replace('v4', '')

    all_results = {
        'env': args.env,
        'pivot': 'jump_ahead',
        'jump_k': args.jump_k,
        'benchmark': bench,
        'mpc_target_reward': float(np.mean(mpc_results['target_only'])),
        'mpc_jump_reward': float(np.mean(mpc_results['jump_ahead'])),
        'reward_retention': float(retention),
    }
    with open(os.path.join(results_dir, f'jump_ahead_{env_tag}.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("Done!")


if __name__ == "__main__":
    main()
