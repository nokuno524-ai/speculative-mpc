"""Round 4: Hybrid Coarse-to-Fine CEM experiment.

1. Train target + jump-ahead models (improved architecture)
2. Sweep K (top-K fine evaluations) for hybrid CEM
3. Compare target-only, jump-only, and hybrid CEM
4. Report wall-clock time, reward retention, full rollouts saved
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time, json, argparse
from collections import deque

from src.main_v2 import DeepRSSM, ReplayBuffer, collect_eps_greedy_data, train_target
from src.jump_ahead import jump_ahead_rollout
from src.jump_ahead_v2 import ImprovedJumpAheadModel, train_jump_ahead_improved
from src.hybrid_cem import HybridCEM, TargetOnlyCEM, JumpOnlyCEM


def run_episode(planner, target, env, device, act_dim, is_discrete):
    """Run one MPC episode, return (total_reward, timing)."""
    obs, _ = env.reset()
    total_reward = 0
    done = False
    ep_timing = {'coarse_ms': 0, 'fine_ms': 0, 'total_ms': 0}

    det, stoch = target.initial_state(1, device)
    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    zero_act = torch.zeros(1, act_dim, device=device)
    det, stoch, _ = target.observe(obs_t, det, stoch, zero_act)

    while not done:
        action_seq, timing = planner.plan(det, stoch, track_time=True)
        action_cont = action_seq[0].cpu().numpy()

        if timing:
            if 'coarse_ms' in timing:
                ep_timing['coarse_ms'] += timing.get('coarse_ms', 0)
                ep_timing['fine_ms'] += timing.get('fine_ms', 0)
            ep_timing['total_ms'] += timing.get('total_ms', timing.get('coarse_ms', 0) + timing.get('fine_ms', 0))

        if is_discrete:
            action_disc = int(action_cont.flat[0] > 0)
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

    return total_reward, ep_timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--n-data-eps', type=int, default=15000)
    parser.add_argument('--target-epochs', type=int, default=600)
    parser.add_argument('--jump-epochs', type=int, default=800)
    parser.add_argument('--jump-k', type=int, default=5)
    parser.add_argument('--n-mpc-eps', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--n-samples', type=int, default=512)
    parser.add_argument('--n-iterations', type=int, default=5)
    parser.add_argument('--top-k-sweep', type=str, default='5,10,20,50')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  Round 4: Hybrid Coarse-to-Fine CEM")
    print(f"  Env: {args.env} | Device: {device} | Jump k={args.jump_k}")
    print(f"{'='*60}")

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    is_discrete = hasattr(env.action_space, 'n')
    act_dim = 1 if is_discrete else env.action_space.shape[0]
    env.close()
    stoch_dim = 30

    # ── Train target ──
    print("\n--- Collecting data ---")
    buffer = collect_eps_greedy_data(args.env, args.n_data_eps, epsilon=0.1)
    print(f"  Buffer size: {len(buffer)} transitions")

    target = DeepRSSM(obs_dim, act_dim, det_dim=200, stoch_dim=stoch_dim).to(device)
    print(f"\n--- Training Target ({args.target_epochs} epochs) ---")
    train_target(target, buffer, n_epochs=args.target_epochs, device=device)

    # ── Train improved jump-ahead ──
    jump_model = ImprovedJumpAheadModel(stoch_dim, act_dim, jump_k=args.jump_k,
                                         hidden_dim=256).to(device)
    n_target = sum(p.numel() for p in target.parameters())
    n_jump = sum(p.numel() for p in jump_model.parameters())
    print(f"\n--- Training Jump-Ahead v2 (k={args.jump_k}, {n_jump:,} params, {n_jump/n_target:.1%} of target) ---")
    train_jump_ahead_improved(target, jump_model, buffer, n_epochs=args.jump_epochs,
                              seq_len=60, device=device)

    # Quality check
    print("\n--- Jump-Ahead Quality Check ---")
    target.eval()
    jump_model.eval()
    with torch.no_grad():
        obs_s, act_s, rew_s = buffer.sample_sequences(256, 40)
        obs_s, act_s = obs_s.to(device), act_s.to(device)
        det, stoch = target.initial_state(256, device)
        target_stochs = []
        for t in range(40):
            det, stoch, _ = target.observe(obs_s[:, t], det, stoch, act_s[:, t])
            target_stochs.append(stoch)
        target_stochs = torch.stack(target_stochs, dim=1)
        k = args.jump_k
        starts = torch.randint(0, 40 - k, (256,), device=device)
        stoch_t = torch.stack([target_stochs[b, s] for b, s in enumerate(starts)])
        stoch_tp1 = torch.stack([target_stochs[b, s + k] for b, s in enumerate(starts)])
        act_chunks = torch.stack([act_s[b, s:s+k] for b, s in enumerate(starts)])
        pred_stoch, _ = jump_model(stoch_t, act_chunks)
        mse = F.mse_loss(pred_stoch, stoch_tp1).item()
        cos_sim = F.cosine_similarity(pred_stoch, stoch_tp1, dim=-1).mean().item()
        print(f"  Jump MSE: {mse:.6f} | Cosine sim: {cos_sim:.4f}")

    # ── Also check coarse ranking quality ──
    print("\n--- Coarse Ranking Quality ---")
    with torch.no_grad():
        N = 128
        H = args.horizon
        det0, stoch0 = target.initial_state(N, device)
        obs_s2, act_s2, _ = buffer.sample_sequences(1, 1)
        obs_t = obs_s2[:, 0].to(device)
        zero_act = torch.zeros(1, act_dim, device=device)
        det0, stoch0 = target.initial_state(N, device)
        # Just use random initial states for ranking test
        actions = torch.randn(N, H, act_dim, device=device).clamp(-1, 1)
        det_exp = det0[:1].expand(N, -1)
        stoch_exp = stoch0[:1].expand(N, -1)

        # Target returns
        _, dets, stochs = target.unroll_imagine(det_exp, stoch_exp, actions, deterministic=True)
        target_returns = target.get_reward(
            dets.reshape(-1, dets.shape[-1]),
            stochs.reshape(-1, stochs.shape[-1])
        ).reshape(N, H).sum(dim=1)

        # Jump-ahead returns
        _, jump_rewards = jump_ahead_rollout(jump_model, stoch_exp, actions)
        jump_returns = jump_rewards.sum(dim=1)

        # Ranking correlation (Spearman - manual)
        def rankdata(x):
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(x)) + 1.0
            return ranks
        tgt_np = target_returns.cpu().numpy()
        jmp_np = jump_returns.cpu().numpy()
        corr = np.corrcoef(rankdata(tgt_np), rankdata(jmp_np))[0, 1]
        print(f"  Spearman correlation (target vs jump returns): {corr:.4f}")

        # Top-K overlap: what fraction of jump's top-20 are in target's top-50?
        for K in [10, 20, 50]:
            jump_topk = set(jump_returns.topk(K).indices.cpu().tolist())
            tgt_top2k = set(target_returns.topk(K * 2).indices.cpu().tolist())
            overlap = len(jump_topk & tgt_top2k) / K
            print(f"  Top-{K} overlap with target top-{K*2}: {overlap:.1%}")

    # ── Baselines ──
    print(f"\n--- Target-Only CEM ({args.n_mpc_eps} episodes) ---")
    planner_tgt = TargetOnlyCEM(target, horizon=args.horizon, n_samples=args.n_samples,
                                 n_iterations=args.n_iterations, action_dim=act_dim)
    env = gym.make(args.env)
    tgt_rewards, tgt_times = [], []
    for ep in range(args.n_mpc_eps):
        r, t = run_episode(planner_tgt, target, env, device, act_dim, is_discrete)
        tgt_rewards.append(r)
        tgt_times.append(t['total_ms'])
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}: reward={r:.1f}, time={t['total_ms']:.0f}ms")
    env.close()
    tgt_mean = np.mean(tgt_rewards)
    tgt_time = np.mean(tgt_times)
    print(f"  Reward: {tgt_mean:.1f} ± {np.std(tgt_rewards):.1f} | Time: {tgt_time:.0f}ms")

    print(f"\n--- Jump-Only CEM ({args.n_mpc_eps} episodes) ---")
    planner_jump = JumpOnlyCEM(jump_model, horizon=args.horizon, n_samples=args.n_samples,
                                n_iterations=args.n_iterations, action_dim=act_dim)
    env = gym.make(args.env)
    jump_rewards, jump_times = [], []
    for ep in range(args.n_mpc_eps):
        r, t = run_episode(planner_jump, target, env, device, act_dim, is_discrete)
        jump_rewards.append(r)
        jump_times.append(t['total_ms'])
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}: reward={r:.1f}, time={t['total_ms']:.0f}ms")
    env.close()
    jump_mean = np.mean(jump_rewards)
    jump_time = np.mean(jump_times)
    print(f"  Reward: {jump_mean:.1f} ± {np.std(jump_rewards):.1f} | Time: {jump_time:.0f}ms")

    # ── Hybrid CEM sweep ──
    top_k_values = [int(k) for k in args.top_k_sweep.split(',')]
    hybrid_results = {}

    for K in top_k_values:
        label = f"Hybrid(top_k={K})"
        print(f"\n--- {label} ({args.n_mpc_eps} episodes) ---")
        planner = HybridCEM(target, jump_model, horizon=args.horizon,
                            n_samples=args.n_samples, n_iterations=args.n_iterations,
                            top_k=K, action_dim=act_dim)

        env = gym.make(args.env)
        h_rewards, h_times, coarse_times, fine_times = [], [], [], []
        for ep in range(args.n_mpc_eps):
            r, t = run_episode(planner, target, env, device, act_dim, is_discrete)
            h_rewards.append(r)
            h_times.append(t['coarse_ms'] + t['fine_ms'])
            coarse_times.append(t['coarse_ms'])
            fine_times.append(t['fine_ms'])
            if (ep + 1) % 5 == 0:
                print(f"  ep {ep+1}: reward={r:.1f}, time={t['coarse_ms']+t['fine_ms']:.0f}ms")
        env.close()

        h_mean = np.mean(h_rewards)
        h_time = np.mean(h_times)
        full_rollouts_per_step = K * args.n_iterations
        target_rollouts_per_step = args.n_samples * args.n_iterations
        rollouts_saved = 1.0 - full_rollouts_per_step / target_rollouts_per_step

        hybrid_results[K] = {
            'reward_mean': h_mean,
            'reward_std': float(np.std(h_rewards)),
            'time_ms': h_time,
            'coarse_ms': np.mean(coarse_times),
            'fine_ms': np.mean(fine_times),
            'retention_vs_target': h_mean / max(tgt_mean, 1),
            'speedup_vs_target': tgt_time / max(h_time, 1),
            'full_rollouts_per_step': full_rollouts_per_step,
            'rollouts_saved_pct': rollouts_saved * 100,
        }
        print(f"  Reward: {h_mean:.1f} ± {np.std(h_rewards):.1f} | "
              f"Time: {h_time:.0f}ms (coarse {np.mean(coarse_times):.0f} + fine {np.mean(fine_times):.0f})")
        print(f"  Retention: {h_mean/max(tgt_mean,1):.1%} | "
              f"Speedup: {tgt_time/max(h_time,1):.2f}x | "
              f"Rollouts saved: {rollouts_saved:.0%}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  ROUND 4 RESULTS — {args.env}")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'Reward':>10} {'Time':>10} {'Retention':>10} {'Speedup':>10} {'Saved':>8}")
    print(f"  {'-'*73}")
    print(f"  {'Target-only':<25} {tgt_mean:>10.1f} {tgt_time:>9.0f}ms {'100.0%':>10} {'1.00x':>10} {'0%':>8}")
    print(f"  {'Jump-only':<25} {jump_mean:>10.1f} {jump_time:>9.0f}ms {jump_mean/max(tgt_mean,1):>9.1%} {tgt_time/max(jump_time,1):>9.2f}x {'100%':>8}")
    for K, res in hybrid_results.items():
        print(f"  {f'Hybrid(K={K})':<25} {res['reward_mean']:>10.1f} {res['time_ms']:>9.0f}ms "
              f"{res['retention_vs_target']:>9.1%} {res['speedup_vs_target']:>9.2f}x "
              f"{res['rollouts_saved_pct']:>7.0f}%")
    print(f"{'='*70}")

    # Save JSON
    results = {
        'env': args.env, 'round': 4, 'jump_k': args.jump_k,
        'jump_mse': mse, 'jump_cosine_sim': cos_sim, 'ranking_corr': corr,
        'horizon': args.horizon, 'n_samples': args.n_samples,
        'n_iterations': args.n_iterations,
        'target_only': {'reward_mean': tgt_mean, 'reward_std': float(np.std(tgt_rewards)), 'time_ms': tgt_time},
        'jump_only': {'reward_mean': jump_mean, 'reward_std': float(np.std(jump_rewards)), 'time_ms': jump_time,
                      'retention': jump_mean / max(tgt_mean, 1)},
        'hybrid_sweep': {str(k): v for k, v in hybrid_results.items()},
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    env_tag = args.env.replace('-', '_')
    with open(os.path.join(results_dir, f'round4_hybrid_{env_tag}.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to results/round4_hybrid_{env_tag}.json")


if __name__ == "__main__":
    main()
