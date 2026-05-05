"""R7: End-to-End CEM Planning Benchmark.

Compares target-only CEM planning vs speculative (draft-based) CEM planning
in an actual control loop on CartPole-v1 and HalfCheetah-v4.

Best draft from r6: L2_D256 (2.30x speedup, 100% acceptance on HalfCheetah).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = '/scratch/qzp4ta/speculative-mpc'

# ── Transformer Draft (same as r5/r6) ─────────────────────────────────────────
class TransformerDraftV2(nn.Module):
    def __init__(self, det_dim=200, stoch_dim=30, act_dim=1,
                 tf_hidden=256, n_heads=4, n_layers=2):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.act_dim = act_dim
        self.act_embed = nn.Linear(act_dim, tf_hidden)
        self.init_proj = nn.Linear(stoch_dim + det_dim, tf_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=n_heads, dim_feedforward=tf_hidden * 4,
            dropout=0.0, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(tf_hidden)
        self.det_head = nn.Sequential(
            nn.Linear(tf_hidden, tf_hidden), nn.GELU(),
            nn.Linear(tf_hidden, det_dim))
        self.stoch_head = nn.Sequential(
            nn.Linear(tf_hidden, tf_hidden), nn.GELU(),
            nn.Linear(tf_hidden, 2 * stoch_dim))

    def forward(self, det_0, stoch_0, actions):
        B, H, _ = actions.shape
        x = self.act_embed(actions) + self.init_proj(
            torch.cat([stoch_0, det_0], -1)).unsqueeze(1)
        mask = nn.Transformer.generate_square_subsequent_mask(H, device=x.device)
        x = self.norm(self.transformer(x, mask=mask))
        dets = self.det_head(x)
        params = self.stoch_head(x)
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        dists = td.Independent(td.Normal(mean, std), 1)
        return dets, mean, dists


# ── CEM Planner variants ─────────────────────────────────────────────────────

class TargetOnlyCEM:
    """Standard CEM: evaluate action sequences via target GRU rollout."""
    def __init__(self, target, horizon=12, n_samples=400, n_iter=5, elite_frac=0.1):
        self.target = target
        self.H = horizon
        self.N = n_samples
        self.n_iter = n_iter
        self.n_elite = max(1, int(n_samples * elite_frac))

    @torch.no_grad()
    def plan(self, det, stoch):
        device = det.device
        act_dim = self.target.act_dim
        mu = torch.zeros(self.H, act_dim, device=device)
        sigma = torch.ones(self.H, act_dim, device=device) * 0.5

        for _ in range(self.n_iter):
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
                self.N, self.H, act_dim, device=device)
            actions = actions.clamp(-1, 1)
            returns = self._eval(det, stoch, actions)
            idx = returns.topk(self.n_elite).indices
            mu = actions[idx].mean(0)
            sigma = actions[idx].std(0) + 1e-6

        return mu[0]  # first action

    def _eval(self, det_0, stoch_0, actions):
        B = actions.shape[0]
        det = det_0.expand(B, -1)
        stoch = stoch_0.expand(B, -1)
        _, dets, stochs = self.target.unroll_imagine(
            det, stoch, actions, deterministic=True)
        rewards = self.target.get_reward(
            dets.reshape(-1, dets.shape[-1]),
            stochs.reshape(-1, stochs.shape[-1])
        ).reshape(B, self.H)
        return rewards.sum(1)


class SpeculativeCEM:
    """Fast CEM: draft predicts all H states in O(1), target reward head scores them."""
    def __init__(self, target, draft, horizon=12, n_samples=400, n_iter=5, elite_frac=0.1):
        self.target = target
        self.draft = draft
        self.H = horizon
        self.N = n_samples
        self.n_iter = n_iter
        self.n_elite = max(1, int(n_samples * elite_frac))

    @torch.no_grad()
    def plan(self, det, stoch):
        device = det.device
        act_dim = self.target.act_dim
        mu = torch.zeros(self.H, act_dim, device=device)
        sigma = torch.ones(self.H, act_dim, device=device) * 0.5

        for _ in range(self.n_iter):
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(
                self.N, self.H, act_dim, device=device)
            actions = actions.clamp(-1, 1)
            returns = self._eval(det, stoch, actions)
            idx = returns.topk(self.n_elite).indices
            mu = actions[idx].mean(0)
            sigma = actions[idx].std(0) + 1e-6

        return mu[0]

    def _eval(self, det_0, stoch_0, actions):
        B = actions.shape[0]
        pred_dets, pred_means, _ = self.draft(det_0.expand(B, -1), stoch_0.expand(B, -1), actions)
        rewards = self.target.get_reward(
            pred_dets.reshape(-1, pred_dets.shape[-1]),
            pred_means.reshape(-1, pred_means.shape[-1])
        ).reshape(B, self.H)
        return rewards.sum(1)


# ── Run episode ───────────────────────────────────────────────────────────────

def run_episode(env, target, planner, max_steps=200):
    """Run one episode using the planner. Returns (reward, planning_times)."""
    obs, _ = env.reset()
    total_reward = 0
    plan_times = []
    device = next(target.parameters()).device

    det = torch.zeros(1, target.det_dim, device=device)
    stoch = torch.zeros(1, target.stoch_dim, device=device)

    for step in range(max_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            obs_embed = target.obs_encoder(obs_t)
            post_params = target.posterior_net(torch.cat([obs_embed, det], -1))
            mean, std = post_params.chunk(2, -1)
            std = F.softplus(std) + 0.1
            stoch = mean  # use mean for deterministic planning

        t0 = time.perf_counter()
        action = planner.plan(det, stoch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        plan_times.append(time.perf_counter() - t0)

        if hasattr(env.action_space, 'n'):
            act_np = int(action.item() > 0)
            obs, reward, term, trunc, _ = env.step(act_np)
            cont_action = torch.FloatTensor([act_np * 2.0 - 1.0]).to(device)
        else:
            act_np = action.cpu().numpy()
            obs, reward, term, trunc, _ = env.step(act_np)
            cont_action = action.unsqueeze(0)

        total_reward += reward

        with torch.no_grad():
            det = target.gru(torch.cat([stoch, cont_action], -1), det)

        if term or trunc:
            break

    return total_reward, plan_times


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', default=['CartPole-v1', 'HalfCheetah-v4'])
    parser.add_argument('--n-episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=12)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    print(f"{'='*70}")
    print(f"  R7: End-to-End CEM Planning Benchmark")
    print(f"  Device: {device} ({gpu_name})")
    print(f"  Episodes: {args.n_episodes} | Horizon: {args.horizon}")
    print(f"{'='*70}")

    all_results = {}

    for env_name in args.envs:
        env_tag = 'CartPole' if 'CartPole' in env_name else 'HalfCheetah'
        act_dim = 1 if 'CartPole' in env_name else 6
        obs_dim = 4 if 'CartPole' in env_name else 17
        max_steps = 500 if 'HalfCheetah' in env_name else 200

        print(f"\n{'='*70}")
        print(f"  Environment: {env_name}")
        print(f"{'='*70}")

        from src.main_v2 import DeepRSSM
        target = DeepRSSM(obs_dim=obs_dim, act_dim=act_dim, det_dim=200, stoch_dim=30)
        ckpt_path = f'{PROJECT}/checkpoints/target_{env_tag}_.pt'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        target.load_state_dict(ckpt)
        target.to(device).eval()
        print(f"  Target loaded")

        draft = TransformerDraftV2(
            det_dim=200, stoch_dim=30, act_dim=act_dim,
            tf_hidden=256, n_layers=2).to(device)
        draft_path = f'{PROJECT}/checkpoints/draft_r6_{env_tag}_L2_D256_S10000.pt'
        if not os.path.exists(draft_path):
            draft_path = f'{PROJECT}/checkpoints/draft_transformer_{env_tag}_best.pt'
        if os.path.exists(draft_path):
            draft.load_state_dict(torch.load(draft_path, map_location=device, weights_only=True))
            print(f"  Draft loaded from {draft_path}")
        else:
            print(f"  No draft at {draft_path}, skipping")
            continue
        draft.eval()

        target_planner = TargetOnlyCEM(target, horizon=args.horizon, n_samples=400, n_iter=5)
        spec_planner = SpeculativeCEM(target, draft, horizon=args.horizon, n_samples=400, n_iter=5)

        env = gym.make(env_name)

        print(f"\n  --- Target-Only CEM ---")
        target_rewards, target_times = [], []
        for ep in range(args.n_episodes):
            reward, ptimes = run_episode(env, target, target_planner, max_steps=max_steps)
            target_rewards.append(reward)
            target_times.extend(ptimes)
            print(f"    ep {ep+1}/{args.n_episodes}: reward={reward:.1f} | plan={np.mean(ptimes)*1000:.1f}ms/step")

        print(f"\n  --- Speculative CEM ---")
        spec_rewards, spec_times = [], []
        for ep in range(args.n_episodes):
            reward, ptimes = run_episode(env, target, spec_planner, max_steps=max_steps)
            spec_rewards.append(reward)
            spec_times.extend(ptimes)
            print(f"    ep {ep+1}/{args.n_episodes}: reward={reward:.1f} | plan={np.mean(ptimes)*1000:.1f}ms/step")

        t_mean_r = np.mean(target_rewards)
        s_mean_r = np.mean(spec_rewards)
        t_mean_t = np.mean(target_times) * 1000
        s_mean_t = np.mean(spec_times) * 1000
        speedup = t_mean_t / max(s_mean_t, 0.01)
        reward_retention = s_mean_r / max(t_mean_r, 0.01) if t_mean_r != 0 else 0

        print(f"\n  Target-only:  reward={t_mean_r:.1f} ± {np.std(target_rewards):.1f}  time={t_mean_t:.1f}ms")
        print(f"  Speculative:  reward={s_mean_r:.1f} ± {np.std(spec_rewards):.1f}  time={s_mean_t:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x | Reward retention: {reward_retention:.1%}")

        all_results[env_name] = {
            'target_reward_mean': float(t_mean_r),
            'target_reward_std': float(np.std(target_rewards)),
            'target_time_ms': float(t_mean_t),
            'spec_reward_mean': float(s_mean_r),
            'spec_reward_std': float(np.std(spec_rewards)),
            'spec_time_ms': float(s_mean_t),
            'speedup': float(speedup),
            'reward_retention': float(reward_retention),
            'n_episodes': args.n_episodes,
            'horizon': args.horizon,
        }
        env.close()

    with open(f'{PROJECT}/results/r7_e2e_benchmark.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot
    if all_results:
        envs = list(all_results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = np.arange(len(envs))
        w = 0.35

        t_r = [all_results[e]['target_reward_mean'] for e in envs]
        s_r = [all_results[e]['spec_reward_mean'] for e in envs]
        t_s = [all_results[e]['target_reward_std'] for e in envs]
        s_s = [all_results[e]['spec_reward_std'] for e in envs]
        axes[0].bar(x-w/2, t_r, w, yerr=t_s, label='Target', color='steelblue', capsize=3)
        axes[0].bar(x+w/2, s_r, w, yerr=s_s, label='Speculative', color='coral', capsize=3)
        axes[0].set_xticks(x); axes[0].set_xticklabels(envs, fontsize=8)
        axes[0].set_ylabel('Reward'); axes[0].set_title('Reward'); axes[0].legend()

        t_t = [all_results[e]['target_time_ms'] for e in envs]
        s_t = [all_results[e]['spec_time_ms'] for e in envs]
        axes[1].bar(x-w/2, t_t, w, label='Target', color='steelblue')
        axes[1].bar(x+w/2, s_t, w, label='Speculative', color='coral')
        axes[1].set_xticks(x); axes[1].set_xticklabels(envs, fontsize=8)
        axes[1].set_ylabel('ms/step'); axes[1].set_title('Planning Time'); axes[1].legend()

        sp = [all_results[e]['speedup'] for e in envs]
        ret = [all_results[e]['reward_retention'] for e in envs]
        axes[2].bar(x-w/2, sp, w, label='Speedup (x)', color='green', alpha=0.7)
        ax3 = axes[2].twinx()
        ax3.bar(x+w/2, ret, w, label='Retention', color='purple', alpha=0.7)
        ax3.set_ylabel('Retention', color='purple')
        axes[2].set_xticks(x); axes[2].set_xticklabels(envs, fontsize=8)
        axes[2].set_ylabel('Speedup (x)', color='green')
        axes[2].set_title('Speedup & Retention')

        fig.suptitle('R7: End-to-End CEM Benchmark', fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{PROJECT}/experiment_results/r7_e2e_benchmark.png', dpi=150)
        print("Plot saved.")


if __name__ == '__main__':
    main()
