"""End-to-end pipeline: train target RSSM → distill MLP draft → CEM benchmark.

Architecture:
  - Target: Deep GRU-based RSSM (sequential O(H) rollout)
  - Draft: Non-autoregressive MLP (single forward pass O(1) → all H states)
  - CEM planner: compares target-only vs speculative planning
  - CartPole-v1 for fast iteration
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
from src.cem_planner import CEMPlanner
from src.acceptance import evaluate_and_accept
from src.parallel_verify import parallel_verify


# ── Deep target RSSM ──────────────────────────────────────────────────────────

class DeepRSSM(RSSM):
    """Target RSSM with deeper networks for richer representations."""
    def __init__(self, obs_dim, act_dim, det_dim=300, stoch_dim=50):
        super().__init__(obs_dim, act_dim, det_dim=det_dim, stoch_dim=stoch_dim,
                         hidden_dim=512)
        # Override with deeper networks
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


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, observations, actions, rewards):
        for t in range(len(rewards)):
            self.buffer.append((
                observations[t].astype(np.float32),
                np.array([actions[t]], dtype=np.float32),
                np.float32(rewards[t]),
                observations[t + 1].astype(np.float32) if t + 1 < len(observations)
                else observations[t].astype(np.float32),
            ))

    def sample_sequences(self, batch_size, seq_len):
        obs_seq, act_seq, rew_seq = [], [], []
        n = len(self.buffer)
        for _ in range(batch_size):
            start = random.randint(0, max(0, n - seq_len - 1))
            o, a, r = [], [], []
            for t in range(start, min(start + seq_len, n)):
                obs, act, reward, _ = self.buffer[t]
                o.append(obs); a.append(act); r.append(reward)
            while len(o) < seq_len:
                o.append(o[-1]); a.append(a[-1]); r.append(np.float32(0.0))
            obs_seq.append(np.stack(o))
            act_seq.append(np.stack(a))
            rew_seq.append(np.array(r, dtype=np.float32))
        return (torch.FloatTensor(np.stack(obs_seq)),
                torch.FloatTensor(np.stack(act_seq)),
                torch.FloatTensor(np.stack(rew_seq)))

    def __len__(self):
        return len(self.buffer)


# ── Training functions ────────────────────────────────────────────────────────

def train_target(model, buffer, n_epochs=300, batch_size=128, seq_len=20,
                 lr=3e-4, device='cpu'):
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
            # Prior at same step (no action needed, use zeros)
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
            print(f"  Target epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f} | "
                  f"Rew: {reward_loss.item():.4f}")

    return model


def train_draft(target, draft, buffer, n_epochs=300, batch_size=128,
                seq_len=20, lr=1e-3, device='cpu'):
    """Distill non-autoregressive draft from trained target.

    Target encodes observations → stochastic states.
    Draft learns: given stoch_0 + actions[1:T], predict stochs[1:T].
    """
    optimizer = optim.Adam(draft.parameters(), lr=lr)
    target.eval()
    draft.train()

    for epoch in range(n_epochs):
        obs, acts, _ = buffer.sample_sequences(batch_size, seq_len)
        obs, acts = obs.to(device), acts.to(device)
        B, T = obs.shape[:2]

        # Get target stochastic states
        with torch.no_grad():
            tgt_det, tgt_stoch = target.initial_state(B, device)
            target_stochs = []
            for t in range(T):
                tgt_det, tgt_stoch, _ = target.observe(
                    obs[:, t], tgt_det, tgt_stoch, acts[:, t])
                target_stochs.append(tgt_stoch)
            target_stochs = torch.stack(target_stochs, dim=1)  # [B, T, stoch_dim]

        optimizer.zero_grad()

        # Draft input: stoch at t=0, actions from t=1..T
        stoch_0 = target_stochs[:, 0]
        draft_actions = acts[:, 1:]  # [B, T-1, 1]
        target_ref = target_stochs[:, 1:]  # [B, T-1, stoch_dim]

        if draft_actions.shape[1] == 0:
            continue

        pred_stochs, pred_dists = draft(stoch_0, draft_actions)

        # MSE alignment loss
        mse_loss = F.mse_loss(pred_stochs, target_ref)

        # KL matching loss (draft dist vs target dist)
        with torch.no_grad():
            # Get target priors for the same steps
            tgt_det2, tgt_stoch2 = target.initial_state(B, device)
            # Re-observe to get to stoch_0 state
            for t in range(1):
                tgt_det2, tgt_stoch2, _ = target.observe(
                    obs[:, t], tgt_det2, tgt_stoch2, acts[:, t])

            target_priors = []
            det, stoch = tgt_det2, tgt_stoch2
            for t in range(T - 1):
                det, stoch, prior = target.imagine_deterministic(
                    det, stoch, draft_actions[:, t])
                target_priors.append(prior)

        if target_priors:
            target_means = torch.stack([p.mean for p in target_priors], dim=1)
            target_stds = torch.stack([p.stddev for p in target_priors], dim=1)
            target_dists = td.Independent(td.Normal(target_means, target_stds), 1)
            kl_loss = td.kl.kl_divergence(target_dists, pred_dists).mean()
            loss = mse_loss + 0.01 * kl_loss
        else:
            loss = mse_loss

        loss.backward()
        nn.utils.clip_grad_norm_(draft.parameters(), 100.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Draft epoch {epoch+1}/{n_epochs} | "
                  f"MSE: {mse_loss.item():.6f} | Loss: {loss.item():.4f}")

    return draft


# ── CEM Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_mpc(target, draft, device, n_episodes=20, horizon=12,
                 n_samples=256, n_iterations=5):
    """Evaluate CEM planning on actual CartPole episodes."""
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]

    planner_tgt = CEMPlanner(target, draft_model=None, horizon=horizon,
                             n_samples=n_samples, n_iterations=n_iterations,
                             action_dim=1, action_low=-1.0, action_high=1.0)
    planner_spec = CEMPlanner(target, draft_model=draft, horizon=horizon,
                              n_samples=n_samples, n_iterations=n_iterations,
                              action_dim=1, action_low=-1.0, action_high=1.0)

    results = {'target_only': [], 'speculative': []}

    for mode, planner in [('target_only', planner_tgt), ('speculative', planner_spec)]:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False

            # Encode initial observation
            det, stoch = target.initial_state(1, device)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            zero_act = torch.zeros(1, 1, device=device)
            det, stoch, _ = target.observe(obs_t, det, stoch, zero_act)

            while not done:
                action_seq = planner.plan_target_only(det, stoch) if mode == 'target_only' \
                    else planner.plan_speculative(det, stoch)
                action_cont = action_seq[0, 0].item()
                action_disc = int(action_cont > 0)

                next_obs, reward, term, trunc, _ = env.step(action_disc)
                total_reward += reward

                # Update state
                obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                act_t = torch.FloatTensor([[action_cont]]).to(device)
                det, stoch, _ = target.observe(obs_t, det, stoch, act_t)

                done = term or trunc

            results[mode].append(total_reward)
            if (ep + 1) % 5 == 0:
                print(f"    {mode} ep {ep+1}: reward={total_reward:.0f}")

    env.close()

    tgt_mean = np.mean(results['target_only'])
    spec_mean = np.mean(results['speculative'])
    print(f"\n  Target-only reward:  {tgt_mean:.1f} ± {np.std(results['target_only']):.1f}")
    print(f"  Speculative reward:  {spec_mean:.1f} ± {np.std(results['speculative']):.1f}")
    print(f"  Reward retention:    {spec_mean / max(tgt_mean, 1):.1%}")

    return results


# ── Benchmark ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_speed(target, draft, device, H=30, batch_size=256, n_trials=100,
                    eps_base=5.0, alpha=0.5):
    """Benchmark rollout speed and acceptance rates."""
    target.eval()
    draft.eval()

    actions = torch.randn(batch_size, H, 1, device=device).clamp(-1, 1)
    det_0, stoch_0 = target.initial_state(batch_size, device)

    # Warmup
    for _ in range(10):
        target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        draft(stoch_0, actions)

    # Target-only sequential rollout
    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None
    sync(); t0 = time.time()
    for _ in range(n_trials):
        _, dets, stochs = target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        target.get_reward(dets.reshape(-1, dets.shape[-1]),
                          stochs.reshape(-1, stochs.shape[-1]))
    sync()
    target_time = (time.time() - t0) / n_trials

    # Draft single-pass
    sync(); t0 = time.time()
    for _ in range(n_trials):
        draft_stochs, draft_dists = draft(stoch_0, actions)
    sync()
    draft_time = (time.time() - t0) / n_trials

    # Speculative: draft + verify
    sync(); t0 = time.time()
    for _ in range(n_trials):
        draft_stochs, draft_dists = draft(stoch_0, actions)
        target_dists, target_dets = parallel_verify(
            target, stoch_0, draft_stochs, actions, det_0)
    sync()
    spec_time = (time.time() - t0) / n_trials

    # Acceptance stats
    draft_stochs, draft_dists = draft(stoch_0, actions)
    target_dists, target_dets = parallel_verify(
        target, stoch_0, draft_stochs, actions, det_0)
    _, accepted_lengths, kl_divs = evaluate_and_accept(
        target_dists, draft_dists, eps_base=eps_base, alpha=alpha)

    return {
        'target_time_ms': target_time * 1000,
        'draft_time_ms': draft_time * 1000,
        'speculative_time_ms': spec_time * 1000,
        'draft_speedup': target_time / max(draft_time, 1e-8),
        'speculative_speedup': target_time / max(spec_time, 1e-8),
        'acceptance_rate': (accepted_lengths.float().mean() / H).item(),
        'mean_kl': kl_divs.mean().item(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Speculative Decoding for World Model Rollouts")
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name()})"
          if device.type == 'cuda' else ""))
    print(f"{'='*60}")

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = 1
    stoch_dim = 50

    # Models
    target = DeepRSSM(obs_dim, act_dim, det_dim=300, stoch_dim=stoch_dim).to(device)
    draft = MLDraft(stoch_dim, act_dim, hidden_dim=256).to(device)

    n_target = sum(p.numel() for p in target.parameters())
    n_draft = sum(p.numel() for p in draft.parameters())
    print(f"\nTarget params: {n_target:,}")
    print(f"Draft params:  {n_draft:,} ({n_draft/n_target:.1%} of target)")

    # Collect data
    buffer = ReplayBuffer(100000)
    print("\n--- Collecting CartPole data ---")
    for ep in range(500):
        obs_list, act_list, rew_list = [env.reset()[0].astype(np.float32)], [], []
        obs = obs_list[0]
        done = False
        while not done:
            action = int(env.action_space.sample())
            cont = np.float32(action * 2.0 - 1.0)
            next_obs, reward, term, trunc, _ = env.step(action)
            act_list.append(cont)
            rew_list.append(np.float32(reward))
            obs_list.append(next_obs.astype(np.float32))
            obs = next_obs
            done = term or trunc
        buffer.add_episode(np.array(obs_list), np.array(act_list), np.array(rew_list))
    print(f"  Collected {len(buffer)} transitions from 500 episodes")
    env.close()

    # Train target
    print("\n--- Training Target RSSM (300 epochs) ---")
    train_target(target, buffer, n_epochs=300, device=device)

    # Save target checkpoint
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(target.state_dict(), os.path.join(ckpt_dir, 'target.pt'))
    print(f"  Saved target to {ckpt_dir}/target.pt")

    # Train draft
    print("\n--- Training Non-Autoregressive Draft MLP (300 epochs) ---")
    train_draft(target, draft, buffer, n_epochs=300, device=device)
    torch.save(draft.state_dict(), os.path.join(ckpt_dir, 'draft.pt'))
    print(f"  Saved draft to {ckpt_dir}/draft.pt")

    # Benchmark speed
    print("\n--- Benchmarking Rollout Speed (H=30) ---")
    bench = benchmark_speed(target, draft, device, H=30, batch_size=256, n_trials=100)
    print(f"  Target sequential:   {bench['target_time_ms']:.2f} ms")
    print(f"  Draft single-pass:   {bench['draft_time_ms']:.2f} ms "
          f"({bench['draft_speedup']:.1f}x faster)")
    print(f"  Speculative (d+v):   {bench['speculative_time_ms']:.2f} ms "
          f"({bench['speculative_speedup']:.2f}x)")
    print(f"  Acceptance rate:     {bench['acceptance_rate']:.1%}")
    print(f"  Mean KL:             {bench['mean_kl']:.4f}")

    # Evaluate MPC
    print("\n--- Evaluating CEM Planning on CartPole ---")
    mpc_results = evaluate_mpc(target, draft, device, n_episodes=20,
                                horizon=12, n_samples=256, n_iterations=5)

    # Save results
    results = {
        **bench,
        'target_params': n_target,
        'draft_params': n_draft,
        'size_ratio': n_draft / n_target,
        'mpc_target_reward': float(np.mean(mpc_results['target_only'])),
        'mpc_speculative_reward': float(np.mean(mpc_results['speculative'])),
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}/results.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Target params:          {n_target:,}")
    print(f"  Draft params:           {n_draft:,} ({n_draft/n_target:.1%})")
    print(f"  Draft speedup:          {bench['draft_speedup']:.1f}x")
    print(f"  Speculative speedup:    {bench['speculative_speedup']:.2f}x")
    print(f"  Acceptance rate:        {bench['acceptance_rate']:.1%}")
    print(f"  MPC target reward:      {results['mpc_target_reward']:.1f}")
    print(f"  MPC speculative reward: {results['mpc_speculative_reward']:.1f}")
    print(f"  Reward retention:       "
          f"{results['mpc_speculative_reward']/max(results['mpc_target_reward'],1):.1%}")
    print(f"{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
