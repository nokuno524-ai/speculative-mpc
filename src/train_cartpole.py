"""End-to-end CartPole-v1: train target → distill draft → benchmark.

Uses a non-autoregressive MLP draft for O(1) rollout speed vs O(H) target.
"""
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
from src.ml_draft import MLDraft


class DeepRSSM(RSSM):
    """Target RSSM with extra-deep networks."""
    def __init__(self, obs_dim, act_dim, det_dim=300, stoch_dim=50):
        super().__init__(obs_dim, act_dim, det_dim=det_dim, stoch_dim=stoch_dim, hidden_dim=512)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(512 + det_dim, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 2 * stoch_dim),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 2 * stoch_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 1),
        )


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, observations, actions, rewards):
        for t in range(len(rewards)):
            self.buffer.append((
                observations[t].astype(np.float32),
                np.array([actions[t]], dtype=np.float32),
                np.float32(rewards[t]),
                observations[t + 1].astype(np.float32) if t + 1 < len(observations) else observations[t].astype(np.float32),
            ))

    def sample_sequences(self, batch_size, seq_len):
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

    def __len__(self):
        return len(self.buffer)


def train_target(model, buffer, n_epochs=200, batch_size=128, seq_len=20, lr=3e-4, device='cpu'):
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
            det, stoch, posterior = model.observe(obs[:, t], det, stoch, acts[:, t])
            _, _, prior = model.imagine(det, stoch, torch.zeros_like(acts[:, t]))
            kl_loss += td.kl.kl_divergence(posterior, prior).mean()
            pred_reward = model.get_reward(det, stoch)
            reward_loss += F.mse_loss(pred_reward, rews[:, t])
        kl_loss = kl_loss / T
        reward_loss = reward_loss / T
        loss = reward_loss + 0.1 * kl_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Target epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f} | Rew: {reward_loss.item():.4f}")


def train_draft(target, draft, buffer, n_epochs=200, batch_size=128, seq_len=20, lr=1e-3, device='cpu'):
    """Train non-autoregressive draft to predict target's stochastic states."""
    optimizer = optim.Adam(draft.parameters(), lr=lr)
    target.eval()
    draft.train()

    for epoch in range(n_epochs):
        obs, acts, _ = buffer.sample_sequences(batch_size, seq_len)
        obs, acts = obs.to(device), acts.to(device)
        B, T = obs.shape[:2]

        optimizer.zero_grad()

        # Get target stochastic states
        with torch.no_grad():
            tgt_det, tgt_stoch = target.initial_state(B, device)
            target_stochs = []
            for t in range(T):
                tgt_det, tgt_stoch, _ = target.observe(obs[:, t], tgt_det, tgt_stoch, acts[:, t])
                target_stochs.append(tgt_stoch)
            target_stochs = torch.stack(target_stochs, dim=1)  # [B, T, stoch_dim]

        # Get initial stoch for draft input (from obs encoding of first obs)
        with torch.no_grad():
            _, init_stoch, _ = target.observe(
                obs[:, 0],
                *target.initial_state(B, device),
                torch.zeros(B, 1, device=device)
            )

        # Draft predicts all T states from init_stoch + actions
        # Use actions from t=1..T, predicting states at t=1..T
        pred_stochs, pred_dists = draft(init_stoch, acts[:, 1:] if T > 1 else acts)

        # Align with target
        if pred_stochs.shape[1] <= target_stochs.shape[1] - 1:
            tgt_ref = target_stochs[:, 1:1 + pred_stochs.shape[1]]
        else:
            tgt_ref = target_stochs[:, 1:]

        align_loss = F.mse_loss(pred_stochs, tgt_ref)

        align_loss.backward()
        nn.utils.clip_grad_norm_(draft.parameters(), 100.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Draft epoch {epoch+1}/{n_epochs} | Alignment: {align_loss.item():.4f}")


@torch.no_grad()
def verify_draft_vs_target(target, draft, det_0, stoch_0, actions, eps_base=2.0, alpha=0.2):
    """Verify draft predictions against target's sequential rollout."""
    from src.acceptance import evaluate_and_accept

    B, H, _ = actions.shape

    # Draft: single forward pass
    draft_stochs, draft_dists = draft(stoch_0, actions)

    # Target: sequential rollout
    det, stoch = det_0, stoch_0
    target_dists_list = []
    for t in range(H):
        det, stoch, prior = target.imagine(det, stoch, actions[:, t])
        target_dists_list.append(prior)

    target_means = torch.stack([d.mean for d in target_dists_list], dim=1)
    target_stds = torch.stack([d.stddev for d in target_dists_list], dim=1)
    target_dists_tensor = td.Independent(td.Normal(target_means, target_stds), 1)

    # Acceptance
    valid_mask, accepted_lengths, kl_divs = evaluate_and_accept(
        target_dists_tensor, draft_dists, eps_base=eps_base, alpha=alpha
    )

    return draft_stochs, valid_mask, accepted_lengths, kl_divs, {
        'avg_acceptance_rate': (accepted_lengths.float().mean() / H).item(),
        'mean_kl': kl_divs.mean().item(),
    }


@torch.no_grad()
def benchmark(target, draft, device, H=30, batch_size=256, n_trials=100):
    target.eval()
    draft.eval()

    actions = torch.randn(batch_size, H, 1, device=device).clamp(-1, 1)
    det_0, stoch_0 = target.initial_state(batch_size, device)

    # Warmup
    for _ in range(10):
        target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        draft(stoch_0, actions)

    # Target: sequential H-step rollout + reward
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    for _ in range(n_trials):
        priors, dets, stochs = target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        for t in range(H):
            target.get_reward(dets[:, t], stochs[:, t])
    torch.cuda.synchronize() if device.type == 'cuda' else None
    target_time = (time.time() - t0) / n_trials

    # Draft: single forward pass
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    for _ in range(n_trials):
        draft_stochs, draft_dists = draft(stoch_0, actions)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    draft_time = (time.time() - t0) / n_trials

    # Speculative: draft + verify (target sequential)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    for _ in range(n_trials):
        verify_draft_vs_target(target, draft, det_0, stoch_0, actions)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    spec_time = (time.time() - t0) / n_trials

    # Stats
    _, _, _, _, stats = verify_draft_vs_target(target, draft, det_0, stoch_0, actions, eps_base=5.0, alpha=0.5)

    return {
        'target_time_ms': target_time * 1000,
        'draft_time_ms': draft_time * 1000,
        'speculative_time_ms': spec_time * 1000,
        'draft_speedup': target_time / max(draft_time, 1e-8),
        'speculative_speedup': target_time / max(spec_time, 1e-8),
        'acceptance_rate': stats['avg_acceptance_rate'],
        'mean_kl': stats['mean_kl'],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Speculative MPC: CartPole-v1 ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = 1
    stoch_dim = 50

    target = DeepRSSM(obs_dim, act_dim, det_dim=300, stoch_dim=stoch_dim).to(device)
    draft = MLDraft(act_dim, stoch_dim, hidden_dim=256).to(device)

    n_target = sum(p.numel() for p in target.parameters())
    n_draft = sum(p.numel() for p in draft.parameters())
    print(f"\nTarget params: {n_target:,}")
    print(f"Draft params:  {n_draft:,} ({n_draft/n_target:.1%} of target)")

    buffer = ReplayBuffer(100000)
    print("\n--- Collecting CartPole data ---")
    for ep in range(300):
        obs_list, act_list, rew_list = [env.reset()[0]], [], []
        obs = obs_list[0]
        done = False
        while not done:
            action = float(env.action_space.sample())
            action_cont = action * 2.0 - 1.0
            next_obs, reward, term, trunc, _ = env.step(int(action))
            act_list.append(action_cont)
            rew_list.append(reward)
            obs_list.append(next_obs)
            obs = next_obs
            done = term or trunc
        buffer.add_episode(np.array(obs_list), np.array(act_list), np.array(rew_list))
    print(f"Collected {len(buffer)} transitions")

    print("\n--- Training Target RSSM ---")
    train_target(target, buffer, n_epochs=200, device=device)

    print("\n--- Training Non-Autoregressive Draft ---")
    train_draft(target, draft, buffer, n_epochs=200, device=device)

    print("\n--- Benchmarking (H=30, batch=256) ---")
    bench = benchmark(target, draft, device, H=30, batch_size=256, n_trials=100)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Target params:          {n_target:,}")
    print(f"Draft params:           {n_draft:,} ({n_draft/n_target:.1%})")
    print(f"Target rollout (seq):   {bench['target_time_ms']:.2f} ms")
    print(f"Draft rollout (O(1)):   {bench['draft_time_ms']:.2f} ms ({bench['draft_speedup']:.1f}x faster)")
    print(f"Speculative (d+verify): {bench['speculative_time_ms']:.2f} ms ({bench['speculative_speedup']:.2f}x)")
    print(f"Acceptance rate:        {bench['acceptance_rate']:.1%}")
    print(f"Mean KL:                {bench['mean_kl']:.4f}")
    print(f"{'='*60}")

    results = {k: v for k, v in bench.items()}
    results['target_params'] = n_target
    results['draft_params'] = n_draft
    results['size_ratio'] = n_draft / n_target
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
