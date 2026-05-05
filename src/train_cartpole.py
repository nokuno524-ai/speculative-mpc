"""End-to-end CartPole-v1: train target → distill draft → benchmark."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import gymnasium as gym
import time, json, random
from collections import deque

from src.rssm import RSSM
from src.draft_model import DraftRSSM
from src.cem_planner import CEMPlanner


# ─── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, observations, actions, rewards):
        """Store an entire episode."""
        for t in range(len(rewards)):
            self.buffer.append((
                observations[t].astype(np.float32),
                np.array([actions[t]], dtype=np.float32),
                np.float32(rewards[t]),
                observations[t + 1].astype(np.float32) if t + 1 < len(observations) else observations[t].astype(np.float32),
            ))

    def sample_sequences(self, batch_size, seq_len):
        """Sample random sequences of length seq_len."""
        obs_seq, act_seq, rew_seq = [], [], []
        n = len(self.buffer)
        for _ in range(batch_size):
            start = random.randint(0, max(0, n - seq_len - 1))
            o, a, r = [], [], []
            for t in range(start, min(start + seq_len, n)):
                obs, act, reward, _ = self.buffer[t]
                o.append(obs)
                a.append(act)
                r.append(reward)
            # Pad if needed
            while len(o) < seq_len:
                o.append(o[-1])
                a.append(a[-1])
                r.append(np.float32(0.0))
            obs_seq.append(np.stack(o))
            act_seq.append(np.stack(a))
            rew_seq.append(np.array(r, dtype=np.float32))
        return (
            torch.FloatTensor(np.stack(obs_seq)),
            torch.FloatTensor(np.stack(act_seq)),
            torch.FloatTensor(np.stack(rew_seq)),
        )

    def sample_single(self, batch_size):
        """Sample single transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, act, rew, _ = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(act)),
            torch.FloatTensor(np.array(rew)),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Training Functions ───────────────────────────────────────────────────────
def train_target(model, buffer, n_epochs=100, batch_size=128, seq_len=20, lr=3e-4, device='cpu'):
    """Train target RSSM on collected data."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        obs, acts, rews = buffer.sample_sequences(batch_size, seq_len)
        obs, acts, rews = obs.to(device), acts.to(device), rews.to(device)
        B, T = obs.shape[:2]

        optimizer.zero_grad()
        det, stoch = model.initial_state(B, device)

        kl_loss = torch.tensor(0.0, device=device)
        reward_loss = torch.tensor(0.0, device=device)

        for t in range(T):
            # Posterior step
            det, stoch, posterior = model.observe(obs[:, t], det, stoch, acts[:, t])
            # Prior (no obs)
            _, _, prior = model.imagine(det, stoch, torch.zeros_like(acts[:, t]))

            kl_loss += td.kl.kl_divergence(posterior, prior).mean()
            pred_reward = model.get_reward(det, stoch)
            reward_loss += F.mse_loss(pred_reward, rews[:, t])

        kl_loss = kl_loss / T
        reward_loss = reward_loss / T
        loss = reward_loss + 0.1 * kl_loss  # KL weight

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"  Target epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f} | Rew: {reward_loss.item():.4f}")


def distill_draft(target, draft, buffer, n_epochs=80, batch_size=128, seq_len=20, lr=1e-3, device='cpu'):
    """Distill target into draft via state alignment."""
    optimizer = optim.Adam(draft.parameters(), lr=lr)
    target.eval()
    draft.train()

    for epoch in range(n_epochs):
        obs, acts, rews = buffer.sample_sequences(batch_size, seq_len)
        obs, acts = obs.to(device), acts.to(device)
        B, T = obs.shape[:2]

        optimizer.zero_grad()

        # Get target stochastic states as reference
        with torch.no_grad():
            tgt_det, tgt_stoch = target.initial_state(B, device)
            target_stochs = []
            for t in range(T):
                tgt_det, tgt_stoch, _ = target.observe(obs[:, t], tgt_det, tgt_stoch, acts[:, t])
                target_stochs.append(tgt_stoch)
            target_stochs = torch.stack(target_stochs, dim=1)  # [B, T, stoch_dim]

        # Run draft, project to target space, align
        drf_det, drf_stoch = draft.initial_state(B, device)
        align_loss = torch.tensor(0.0, device=device)

        for t in range(T):
            drf_det, drf_stoch, _ = draft.imagine(drf_det, drf_stoch, acts[:, t])
            projected = draft.project_to_target(drf_stoch)
            align_loss += F.mse_loss(projected, target_stochs[:, t])

        align_loss = align_loss / T
        align_loss.backward()
        nn.utils.clip_grad_norm_(draft.parameters(), 100.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Distill epoch {epoch+1}/{n_epochs} | Alignment Loss: {align_loss.item():.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Speculative MPC: CartPole-v1 ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]  # 4
    act_dim = 1  # continuous action (maps discrete to [-1, 1])

    # Model configs
    target_det_dim, target_stoch_dim = 200, 30
    draft_det_dim, draft_stoch_dim = 50, 16

    target = RSSM(obs_dim, act_dim, det_dim=target_det_dim, stoch_dim=target_stoch_dim).to(device)
    draft = DraftRSSM(act_dim, det_dim=draft_det_dim, stoch_dim=draft_stoch_dim,
                      target_stoch_dim=target_stoch_dim).to(device)

    n_target = sum(p.numel() for p in target.parameters())
    n_draft = sum(p.numel() for p in draft.parameters())
    print(f"\nTarget params: {n_target:,}")
    print(f"Draft params:  {n_draft:,} ({n_draft/n_target:.1%} of target)")

    # ─── Collect data ─────────────────────────────────────────────────────
    buffer = ReplayBuffer(100000)
    print("\n--- Collecting CartPole data ---")
    for ep in range(200):
        obs_list, act_list, rew_list = [env.reset()[0]], [], []
        obs = obs_list[0]
        done = False
        while not done:
            # Map discrete action to continuous
            action = float(env.action_space.sample())  # 0 or 1
            action_cont = action * 2.0 - 1.0  # map to [-1, 1]
            next_obs, reward, term, trunc, _ = env.step(int(action))
            act_list.append(action_cont)
            rew_list.append(reward)
            obs_list.append(next_obs)
            obs = next_obs
            done = term or trunc
        buffer.add_episode(
            np.array(obs_list),
            np.array(act_list),
            np.array(rew_list),
        )
    print(f"Collected {len(buffer)} transitions from 200 episodes")

    # ─── Train target ─────────────────────────────────────────────────────
    print("\n--- Training Target RSSM ---")
    train_target(target, buffer, n_epochs=100, device=device)

    # ─── Distill draft ────────────────────────────────────────────────────
    print("\n--- Distilling Draft RSSM ---")
    distill_draft(target, draft, buffer, n_epochs=80, device=device)

    # ─── Benchmark ────────────────────────────────────────────────────────
    print("\n--- Benchmarking ---")
    target.eval()
    draft.eval()

    planner = CEMPlanner(target, draft, horizon=12, n_samples=64, action_dim=act_dim)
    det_0, stoch_0 = target.initial_state(64, device)

    bench = planner.benchmark(det_0, stoch_0, n_trials=50, batch_size=64)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Target params:      {n_target:,}")
    print(f"Draft params:       {n_draft:,} ({n_draft/n_target:.1%})")
    print(f"Target time:        {bench['target_time_ms']:.2f} ms")
    print(f"Speculative time:   {bench['speculative_time_ms']:.2f} ms")
    print(f"Speedup:            {bench['speedup']:.2f}x")
    print(f"Acceptance rate:    {bench['acceptance_rate']:.1%}")
    print(f"Mean KL:            {bench['mean_kl']:.4f}")
    print(f"{'='*50}")

    # Save results
    results = {
        'target_params': n_target,
        'draft_params': n_draft,
        'size_ratio': n_draft / n_target,
        'target_time_ms': bench['target_time_ms'],
        'speculative_time_ms': bench['speculative_time_ms'],
        'speedup': bench['speedup'],
        'acceptance_rate': bench['acceptance_rate'],
        'mean_kl': bench['mean_kl'],
    }
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
