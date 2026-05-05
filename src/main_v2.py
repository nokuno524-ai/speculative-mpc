"""Round 2: Improved Speculative MPC pipeline addressing Gemini review feedback.

Key changes from v1:
  1. Epsilon-greedy data collection (not random) for better target training
  2. KL threshold sweep to find real acceptance boundary (target 70-90%)
  3. HalfCheetah-v4 support (harder continuous env)
  4. Proper evaluation with reward retention target >95%
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
import time, json, random, argparse
from collections import deque

from src.rssm import RSSM
from src.ml_draft import MLDraft
from src.ml_draft_v2 import CausalConvDraft
from src.cem_planner import CEMPlanner
from src.acceptance import evaluate_and_accept
from src.parallel_verify import parallel_verify


# ── Deep target RSSM ──────────────────────────────────────────────────────────

class DeepRSSM(RSSM):
    def __init__(self, obs_dim, act_dim, det_dim=200, stoch_dim=30):
        super().__init__(obs_dim, act_dim, det_dim=det_dim, stoch_dim=stoch_dim,
                         hidden_dim=256)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(256 + det_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 1),
        )


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def add_episode(self, observations, actions, rewards):
        for t in range(len(rewards)):
            next_obs = observations[t + 1] if t + 1 < len(observations) else observations[t]
            act = np.array(actions[t], dtype=np.float32).flatten()
            self.buffer.append((
                observations[t].astype(np.float32),
                act,
                np.float32(rewards[t]),
                next_obs.astype(np.float32),
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


# ── Epsilon-Greedy Data Collection ───────────────────────────────────────────

def collect_eps_greedy_data(env_name, n_episodes, epsilon=0.1, buffer=None):
    """Collect data with epsilon-greedy policy using a simple heuristic.

    For CartPole: heuristic = push toward center (angle-based).
    For HalfCheetah: heuristic = move forward (simple sinusoidal).
    """
    env = gym.make(env_name)
    if buffer is None:
        buffer = ReplayBuffer(200000)

    is_discrete = hasattr(env.action_space, 'n')
    obs_dim = env.observation_space.shape[0]

    total_reward = 0
    for ep in range(n_episodes):
        obs_list = [env.reset()[0].astype(np.float32)]
        act_list, rew_list = [], []
        obs = obs_list[0]
        done = False
        ep_reward = 0

        while not done:
            if random.random() < epsilon:
                # Random exploration
                if is_discrete:
                    action = env.action_space.sample()
                    cont_action = np.float32(action * 2.0 - 1.0)
                else:
                    action = env.action_space.sample()
                    cont_action = action.astype(np.float32)
            else:
                # Greedy heuristic
                cont_action = _heuristic_action(env_name, obs, is_discrete)

            if is_discrete:
                action = int(cont_action > 0) if np.ndim(cont_action) == 0 else int(cont_action)
                next_obs, reward, term, trunc, _ = env.step(action)
                cont_action = np.array([action * 2.0 - 1.0], dtype=np.float32)
            else:
                cont_action = np.asarray(cont_action, dtype=np.float32)
                next_obs, reward, term, trunc, _ = env.step(cont_action)

            act_list.append(cont_action)
            rew_list.append(np.float32(reward))
            obs_list.append(next_obs.astype(np.float32))
            obs = next_obs
            done = term or trunc
            ep_reward += reward

        total_reward += ep_reward
        buffer.add_episode(np.array(obs_list), np.array(act_list), np.array(rew_list))

        if (ep + 1) % 1000 == 0:
            avg = total_reward / (ep + 1)
            print(f"  Episode {ep+1}/{n_episodes} | avg reward: {avg:.1f} | buffer: {len(buffer)}")

    env.close()
    print(f"  Collected {len(buffer)} total transitions, avg reward: {total_reward/n_episodes:.1f}")
    return buffer


def _heuristic_action(env_name, obs, is_discrete):
    """Simple heuristic for epsilon-greedy policy."""
    if "CartPole" in env_name:
        # Push cart toward center based on angle and position
        angle = obs[2] if len(obs) > 2 else 0
        pos = obs[0] if len(obs) > 0 else 0
        action = 1.0 if angle + pos * 0.3 > 0 else -1.0
        return np.float32(action)
    elif "HalfCheetah" in env_name:
        # Simple forward-biased sinusoidal gait
        return np.random.uniform(0.3, 1.0, size=6).astype(np.float32)
    else:
        if is_discrete:
            return np.float32(0)
        return np.zeros(1, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_target(model, buffer, n_epochs=500, batch_size=128, seq_len=20,
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

        if (epoch + 1) % 100 == 0:
            print(f"  Target epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f} | "
                  f"Rew: {reward_loss.item():.4f}")

    return model


def train_draft(target, draft, buffer, n_epochs=500, batch_size=128,
                seq_len=20, lr=1e-3, device='cpu'):
    """Distill draft from trained target.

    v3: Train draft to predict TARGET PRIOR states (not posterior).
    The posterior depends on observations which the draft doesn't have.
    The prior depends only on GRU hidden state + previous stochastic state.
    """
    optimizer = optim.Adam(draft.parameters(), lr=lr)
    target.eval()
    draft.train()

    for epoch in range(n_epochs):
        obs, acts, rews = buffer.sample_sequences(batch_size, seq_len)
        obs, acts, rews = obs.to(device), acts.to(device), rews.to(device)
        B, T = obs.shape[:2]

        # Get target PRIOR stochastic states (what the draft should predict)
        # First observe to get initial state
        with torch.no_grad():
            det, stoch = target.initial_state(B, device)
            det, stoch, _ = target.observe(obs[:, 0], det, stoch, acts[:, 0])

            # Then imagine forward to get prior predictions
            prior_stochs = []
            prior_dets = []
            for t in range(1, T):
                det, stoch, prior = target.imagine_deterministic(
                    det, stoch, acts[:, t])
                prior_stochs.append(stoch)  # prior mean
                prior_dets.append(det)

            if not prior_stochs:
                continue
            prior_stochs = torch.stack(prior_stochs, dim=1)  # [B, T-1, stoch_dim]
            prior_dets = torch.stack(prior_dets, dim=1)

        optimizer.zero_grad()

        stoch_0 = prior_stochs[:, 0] if prior_stochs.shape[1] > 0 else stoch
        # Actually use the posterior at t=0 as initial state
        det0, stoch0 = target.initial_state(B, device)
        det0, stoch0, _ = target.observe(obs[:, 0], det0, stoch0, acts[:, 0])
        stoch_0 = stoch0.detach()

        draft_actions = acts[:, 1:]  # [B, T-1, act_dim]
        if draft_actions.shape[1] == 0:
            continue

        pred_stochs, pred_dists = draft(stoch_0, draft_actions)

        # Align with target prior predictions
        mse_loss = F.mse_loss(pred_stochs, prior_stochs.detach())

        loss = mse_loss
        loss.backward()
        nn.utils.clip_grad_norm_(draft.parameters(), 100.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Draft epoch {epoch+1}/{n_epochs} | "
                  f"MSE: {mse_loss.item():.6f}")

    return draft


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_mpc(target, draft, device, env_name="CartPole-v1", n_episodes=20,
                 horizon=12, n_samples=256, n_iterations=5):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    is_discrete = hasattr(env.action_space, 'n')

    act_dim = env.action_space.shape[0] if not is_discrete else 1
    action_low = -1.0 if is_discrete else float(env.action_space.low[0])
    action_high = 1.0 if is_discrete else float(env.action_space.high[0])

    planner_tgt = CEMPlanner(target, draft_model=None, horizon=horizon,
                             n_samples=n_samples, n_iterations=n_iterations,
                             action_dim=act_dim, action_low=action_low, action_high=action_high)
    planner_spec = CEMPlanner(target, draft_model=draft, horizon=horizon,
                              n_samples=n_samples, n_iterations=n_iterations,
                              action_dim=act_dim, action_low=action_low, action_high=action_high)

    results = {'target_only': [], 'speculative': []}

    for mode, planner in [('target_only', planner_tgt), ('speculative', planner_spec)]:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False

            det, stoch = target.initial_state(1, device)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            zero_act = torch.zeros(1, act_dim, device=device)
            det, stoch, _ = target.observe(obs_t, det, stoch, zero_act)

            while not done:
                action_seq = planner.plan_target_only(det, stoch) if mode == 'target_only' \
                    else planner.plan_speculative(det, stoch)
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
                act_t = torch.FloatTensor(act_flat).unsqueeze(0).to(device)  # [1, act_dim]
                det, stoch, _ = target.observe(obs_t, det, stoch, act_t)
                done = term or trunc

            results[mode].append(total_reward)
            if (ep + 1) % 5 == 0:
                print(f"    {mode} ep {ep+1}: reward={total_reward:.1f}")

    env.close()

    tgt_mean = np.mean(results['target_only'])
    spec_mean = np.mean(results['speculative'])
    retention = spec_mean / max(tgt_mean, 1)
    print(f"\n  Target-only reward:  {tgt_mean:.1f} ± {np.std(results['target_only']):.1f}")
    print(f"  Speculative reward:  {spec_mean:.1f} ± {np.std(results['speculative']):.1f}")
    print(f"  Reward retention:    {retention:.1%}")

    return results, retention


# ── KL Threshold Sweep ────────────────────────────────────────────────────────

def sweep_kl_thresholds(target, draft, device, buffer, H=30, batch_size=256, n_trials=100):
    """Sweep epsilon_base and alpha to find acceptance rate in 70-90% range."""
    target.eval()
    draft.eval()

    actions = torch.randn(batch_size, H, draft.act_dim, device=device).clamp(-1, 1)
    det_0, stoch_0 = target.initial_state(batch_size, device)

    # First measure actual KL to set sweep range
    with torch.no_grad():
        draft_stochs_t, draft_dists_t = draft(stoch_0[:64], actions[:64])
        target_dists_t, _ = parallel_verify(
            target, stoch_0[:64], draft_stochs_t, actions[:64], det_0[:64])
        sample_kl = td.kl.kl_divergence(target_dists_t, draft_dists_t).mean().item()
    print(f"  Sample mean KL: {sample_kl:.4f}")

    # Set sweep range based on actual KL
    eps_values = sorted(set(
        [0.001, 0.01, 0.05, 0.1] +
        [round(sample_kl * f, 4) for f in [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]]
    ))
    alpha_values = [0.0, 0.001, 0.005, 0.01]

    # Compute draft predictions and target verification once
    with torch.no_grad():
        draft_stochs, draft_dists = draft(stoch_0, actions)
        target_dists, target_dets = parallel_verify(
            target, stoch_0, draft_stochs, actions, det_0)

    results = []
    print("\n--- KL Threshold Sweep ---")
    print(f"{'eps_base':>10} {'alpha':>8} {'accept_rate':>12} {'mean_kl':>10} {'median_acc_len':>15}")
    print("-" * 60)

    for eps in eps_values:
        for alpha in alpha_values:
            with torch.no_grad():
                _, accepted_lengths, kl_divs = evaluate_and_accept(
                    target_dists, draft_dists, eps_base=eps, alpha=alpha)
                acc_rate = (accepted_lengths.float().mean() / H).item()
                mean_kl = kl_divs.mean().item()
                median_len = accepted_lengths.float().median().item()

            row = {'eps_base': eps, 'alpha': alpha, 'acceptance_rate': acc_rate,
                   'mean_kl': mean_kl, 'median_accepted_length': median_len}
            results.append(row)
            print(f"{eps:>10.3f} {alpha:>8.3f} {acc_rate:>12.1%} {mean_kl:>10.4f} {median_len:>15.0f}")

    # Find sweet spot: acceptance rate between 0.70 and 0.90
    sweet = [r for r in results if 0.70 <= r['acceptance_rate'] <= 0.90]
    if sweet:
        best = min(sweet, key=lambda r: abs(r['acceptance_rate'] - 0.80))
        print(f"\n  → Best threshold: eps_base={best['eps_base']}, alpha={best['alpha']}, "
              f"acceptance={best['acceptance_rate']:.1%}")
    else:
        # Find closest to 80%
        best = min(results, key=lambda r: abs(r['acceptance_rate'] - 0.80))
        print(f"\n  → Closest to 80%: eps_base={best['eps_base']}, alpha={best['alpha']}, "
              f"acceptance={best['acceptance_rate']:.1%}")

    return results, best


# ── Benchmark ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_speed(target, draft, device, act_dim=1, H=30, batch_size=256,
                    n_trials=100, eps_base=0.01, alpha=0.005):
    target.eval()
    draft.eval()

    actions = torch.randn(batch_size, H, act_dim, device=device).clamp(-1, 1)
    det_0, stoch_0 = target.initial_state(batch_size, device)

    # Warmup
    for _ in range(10):
        target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        draft(stoch_0, actions)

    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None

    # Target-only
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
        parallel_verify(target, stoch_0, draft_stochs, actions, det_0)
    sync()
    spec_time = (time.time() - t0) / n_trials

    # Acceptance with chosen thresholds
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Environment name (CartPole-v1 or HalfCheetah-v4)')
    parser.add_argument('--n-episodes', type=int, default=10000,
                        help='Number of episodes for data collection')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon for epsilon-greedy data collection')
    parser.add_argument('--target-epochs', type=int, default=500)
    parser.add_argument('--draft-epochs', type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Speculative MPC — Round 2 (Gemini Review Feedback)")
    print(f"  Env: {args.env} | Device: {device}" +
          (f" ({torch.cuda.get_device_name()})" if device.type == 'cuda' else ""))
    print(f"{'='*60}")

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    is_discrete = hasattr(env.action_space, 'n')
    act_dim = 1 if is_discrete else env.action_space.shape[0]
    stoch_dim = 30
    env.close()

    # Models
    target = DeepRSSM(obs_dim, act_dim, det_dim=200, stoch_dim=stoch_dim).to(device)
    draft = CausalConvDraft(stoch_dim, act_dim, hidden_dim=128, n_layers=4).to(device)

    n_target = sum(p.numel() for p in target.parameters())
    n_draft = sum(p.numel() for p in draft.parameters())
    print(f"\nTarget params: {n_target:,}")
    print(f"Draft params:  {n_draft:,} ({n_draft/n_target:.1%} of target)")

    # ── Step 1: Collect epsilon-greedy data ──────────────────────────────────
    print(f"\n--- Collecting {args.n_episodes} episodes (ε={args.epsilon}) ---")
    buffer = collect_eps_greedy_data(args.env, args.n_episodes, epsilon=args.epsilon)

    # ── Step 2: Train target RSSM ────────────────────────────────────────────
    print(f"\n--- Training Target RSSM ({args.target_epochs} epochs) ---")
    train_target(target, buffer, n_epochs=args.target_epochs, device=device)

    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    env_tag = args.env.replace('-', '_').replace('v1', '').replace('v4', '')
    torch.save(target.state_dict(), os.path.join(ckpt_dir, f'target_{env_tag}.pt'))
    print(f"  Saved target to {ckpt_dir}/target_{env_tag}.pt")

    # ── Step 3: Train draft MLP ──────────────────────────────────────────────
    print(f"\n--- Training Draft MLP ({args.draft_epochs} epochs) ---")
    train_draft(target, draft, buffer, n_epochs=args.draft_epochs, device=device)
    torch.save(draft.state_dict(), os.path.join(ckpt_dir, f'draft_{env_tag}.pt'))
    print(f"  Saved draft to {ckpt_dir}/draft_{env_tag}.pt")

    # ── Step 4: KL threshold sweep ───────────────────────────────────────────
    sweep_results, best_threshold = sweep_kl_thresholds(
        target, draft, device, buffer, H=30, batch_size=256)

    # ── Step 5: Benchmark with best threshold ────────────────────────────────
    print(f"\n--- Benchmarking (H=30, eps={best_threshold['eps_base']}, "
          f"alpha={best_threshold['alpha']}) ---")
    bench = benchmark_speed(target, draft, device, act_dim=act_dim,
                            eps_base=best_threshold['eps_base'],
                            alpha=best_threshold['alpha'])
    print(f"  Target sequential:       {bench['target_time_ms']:.2f} ms")
    print(f"  Draft single-pass:       {bench['draft_time_ms']:.2f} ms "
          f"({bench['draft_speedup']:.1f}x)")
    print(f"  Speculative (draft+ver): {bench['speculative_time_ms']:.2f} ms "
          f"({bench['speculative_speedup']:.1f}x)")
    print(f"  Acceptance rate:         {bench['acceptance_rate']:.1%}")
    print(f"  Mean KL:                 {bench['mean_kl']:.4f}")

    # ── Step 6: MPC evaluation ───────────────────────────────────────────────
    print(f"\n--- Evaluating CEM Planning on {args.env} ---")
    mpc_results, retention = evaluate_mpc(
        target, draft, device, env_name=args.env, n_episodes=20,
        horizon=12, n_samples=256, n_iterations=5)

    # ── Save results ─────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        'env': args.env,
        'data_collection': {'n_episodes': args.n_episodes, 'epsilon': args.epsilon},
        'model_sizes': {'target_params': n_target, 'draft_params': n_draft,
                        'size_ratio': n_draft / n_target},
        'benchmark': bench,
        'best_threshold': best_threshold,
        'kl_sweep': sweep_results,
        'mpc_target_reward': float(np.mean(mpc_results['target_only'])),
        'mpc_speculative_reward': float(np.mean(mpc_results['speculative'])),
        'reward_retention': float(retention),
    }

    with open(os.path.join(results_dir, f'results_{env_tag}.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {args.env}")
    print(f"{'='*60}")
    print(f"  Data: {args.n_episodes} eps, ε={args.epsilon}")
    print(f"  Target params:          {n_target:,}")
    print(f"  Draft params:           {n_draft:,} ({n_draft/n_target:.1%})")
    print(f"  Draft speedup:          {bench['draft_speedup']:.1f}x")
    print(f"  Speculative speedup:    {bench['speculative_speedup']:.1f}x")
    print(f"  Best KL threshold:      eps={best_threshold['eps_base']}, "
          f"alpha={best_threshold['alpha']}")
    print(f"  Acceptance rate:        {bench['acceptance_rate']:.1%}")
    print(f"  MPC target reward:      {all_results['mpc_target_reward']:.1f}")
    print(f"  MPC speculative reward: {all_results['mpc_speculative_reward']:.1f}")
    print(f"  Reward retention:       {retention:.1%}")
    print(f"{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
