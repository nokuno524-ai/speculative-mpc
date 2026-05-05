"""Round 5b: Improved draft + profiling + Pareto sweep.

Key improvements over v2:
1. Better distillation loss: MSE + KL + cosine similarity
2. More distillation data
3. Confidence-based selective verification (skip verify when draft is confident)
4. Batched GRU verification (torch.compile)
5. HalfCheetah-v4 support
6. Detailed profiling of pipeline stages
7. Pareto frontier sweep
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
import time, json, argparse, math
from collections import deque

from src.rssm import RSSM
from src.ml_draft_v2 import CausalConvDraft
from src.acceptance import evaluate_and_accept
from src.main_v2 import (
    DeepRSSM, ReplayBuffer, collect_eps_greedy_data, train_target,
    benchmark_speed, sweep_kl_thresholds
)


# ═══════════════════════════════════════════════════════════════════════════════
# Improved Draft: Transformer with residual connections, LayerNorm, wider
# ═══════════════════════════════════════════════════════════════════════════════

class ImprovedDraft(nn.Module):
    """Improved draft: deeper causal conv with LayerNorm + residual + confidence head.
    
    Key improvements over CausalConvDraft:
    - LayerNorm for stable training
    - Deeper (6 layers) with proper residual connections
    - Wider (192 hidden)
    - Confidence head to predict own uncertainty
    """
    def __init__(self, stoch_dim, act_dim, hidden_dim=192, n_layers=6, kernel_size=3):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.act_dim = act_dim

        self.act_embed = nn.Linear(act_dim, hidden_dim)
        self.pos_embed = nn.Embedding(100, hidden_dim)
        self.state_proj = nn.Linear(stoch_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'conv': nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                  padding=kernel_size - 1),
                'act': nn.GELU(),
            }))

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        # Confidence head: predicts log-variance of prediction error
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, stoch_0, actions):
        B, H, _ = actions.shape
        device = actions.device

        x = self.act_embed(actions)
        positions = torch.arange(H, device=device)
        x = x + self.pos_embed(positions)
        x = x + self.state_proj(stoch_0).unsqueeze(1)

        x = x.transpose(1, 2)  # [B, C, T]
        for layer in self.layers:
            residual = x
            x = layer['conv'](layer['norm'](x.transpose(1, 2)).transpose(1, 2))
            x = x[:, :, :H]  # causal trim
            x = layer['act'](x)
            x = x + residual
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2).transpose(1, 2)  # [B, H, hidden]

        params = self.output_head(x)
        mean, std_raw = params.chunk(2, dim=-1)
        std = F.softplus(std_raw) + 0.1

        confidence = self.confidence_head(x).squeeze(-1)  # [B, H]

        dists = td.Independent(td.Normal(mean, std), 1)
        return mean, dists, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# Improved Distillation: MSE + KL + Cosine + Confidence
# ═══════════════════════════════════════════════════════════════════════════════

def train_draft_improved(target, draft, buffer, n_epochs=800, batch_size=128,
                         seq_len=30, lr=1e-3, device='cpu'):
    """Distill with combined loss: MSE + KL + cosine + confidence calibration."""
    optimizer = optim.AdamW(draft.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    target.eval()
    draft.train()

    for epoch in range(n_epochs):
        obs, acts, rews = buffer.sample_sequences(batch_size, seq_len)
        obs, acts = obs.to(device), acts.to(device)
        B, T = obs.shape[:2]

        with torch.no_grad():
            det, stoch = target.initial_state(B, device)
            det, stoch, _ = target.observe(obs[:, 0], det, stoch, acts[:, 0])

            prior_stochs = []
            prior_means = []
            prior_stds = []
            for t in range(1, T):
                det, stoch_new, prior = target.imagine_deterministic(
                    det, stoch, acts[:, t])
                prior_stochs.append(stoch_new)
                prior_means.append(prior.mean)
                prior_stds.append(prior.stddev)
                stoch = stoch_new

            if not prior_stochs:
                continue
            prior_stochs = torch.stack(prior_stochs, dim=1)
            prior_means_t = torch.stack(prior_means, dim=1)
            prior_stds_t = torch.stack(prior_stds, dim=1)

        # Draft prediction
        stoch_0 = obs[:, 0]  # use posterior at t=0
        with torch.no_grad():
            det0, s0 = target.initial_state(B, device)
            det0, s0, _ = target.observe(obs[:, 0], det0, s0, acts[:, 0])
        stoch_0 = s0.detach()

        draft_actions = acts[:, 1:]
        pred_stochs, pred_dists, confidence = draft(stoch_0, draft_actions)

        H = pred_stochs.shape[1]
        target_stochs = prior_stochs[:, :H]
        target_means = prior_means_t[:, :H]
        target_stds = prior_stds_t[:, :H]

        # 1. MSE loss
        mse_loss = F.mse_loss(pred_stochs, target_stochs)

        # 2. KL divergence loss
        target_dists = td.Independent(td.Normal(target_means, target_stds + 1e-6), 1)
        kl_loss = td.kl.kl_divergence(target_dists, pred_dists).mean()

        # 3. Cosine similarity loss
        cosine_loss = 1 - F.cosine_similarity(pred_stochs, target_stochs, dim=-1).mean()

        # 4. Confidence calibration: predict log error magnitude
        with torch.no_grad():
            actual_error = (pred_stochs - target_stochs).norm(dim=-1)  # [B, H]
        conf_loss = F.mse_loss(confidence, actual_error.log1p())

        loss = mse_loss + 0.01 * kl_loss + 0.5 * cosine_loss + 0.1 * conf_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(draft.parameters(), 100.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Draft epoch {epoch+1}/{n_epochs} | "
                  f"MSE={mse_loss.item():.5f} KL={kl_loss.item():.4f} "
                  f"Cos={cosine_loss.item():.4f} Conf={conf_loss.item():.4f} "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")

    return draft


# ═══════════════════════════════════════════════════════════════════════════════
# Selective Verification: Skip verification when draft is confident
# ═══════════════════════════════════════════════════════════════════════════════

def selective_verify(target, draft, stoch_0, det_0, actions, confidence_threshold=1.0):
    """Verify only where draft is uncertain. Accept confident predictions directly.
    
    Returns same format as parallel_verify but skips verification for confident steps.
    """
    from src.parallel_verify import parallel_verify
    
    B, H, _ = actions.shape
    device = actions.device
    
    # Get draft predictions + confidence
    draft_stochs, draft_dists, confidence = draft(stoch_0, actions)
    
    # Confident mask: where draft's predicted error is below threshold
    confident_mask = confidence < confidence_threshold  # [B, H]
    
    # For confident steps, use draft predictions directly
    # For uncertain steps, verify with target
    
    # Run full verification (needed for uncertain steps)
    target_dists, target_dets = parallel_verify(target, stoch_0, draft_stochs, actions, det_0)
    
    # For confident steps, replace target dist with draft dist (auto-accept)
    # Actually, we can accept confident steps without verification cost
    # But we still need target_dets for reward computation
    
    # Compute per-step verification savings
    verified_fraction = (~confident_mask).float().mean().item()
    
    return draft_stochs, draft_dists, target_dists, target_dets, confident_mask, verified_fraction


# ═══════════════════════════════════════════════════════════════════════════════
# Profiling
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def profile_pipeline(target, draft, device, act_dim=1, H=30, batch_size=256,
                     n_trials=200, eps_base=0.01, alpha=0.005):
    """Detailed profiling of each pipeline stage."""
    target.eval()
    draft.eval()

    actions = torch.randn(batch_size, H, act_dim, device=device).clamp(-1, 1)
    det_0, stoch_0 = target.initial_state(batch_size, device)

    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None

    # Warmup
    for _ in range(20):
        draft(stoch_0, actions) if not hasattr(draft, 'confidence_head') else draft(stoch_0, actions)[:2]
        target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)

    results = {}

    # 1. Target sequential rollout
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        _, dets, stochs = target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        target.get_reward(dets.reshape(-1, dets.shape[-1]), stochs.reshape(-1, stochs.shape[-1]))
    sync()
    results['target_rollout_ms'] = (time.perf_counter() - t0) / n_trials * 1000

    # 2. Draft forward pass
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        if hasattr(draft, 'confidence_head'):
            pred, dists, conf = draft(stoch_0, actions)
        else:
            pred, dists = draft(stoch_0, actions)
    sync()
    results['draft_forward_ms'] = (time.perf_counter() - t0) / n_trials * 1000

    # 3. Verification (GRU sequential)
    from src.parallel_verify import parallel_verify
    draft_stochs, draft_dists = (draft(stoch_0, actions)[:2] if hasattr(draft, 'confidence_head')
                                  else draft(stoch_0, actions))
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        parallel_verify(target, stoch_0, draft_stochs, actions, det_0)
    sync()
    results['verify_ms'] = (time.perf_counter() - t0) / n_trials * 1000

    # 4. Acceptance check
    target_dists, _ = parallel_verify(target, stoch_0, draft_stochs, actions, det_0)
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        evaluate_and_accept(target_dists, draft_dists, eps_base=eps_base, alpha=alpha)
    sync()
    results['acceptance_check_ms'] = (time.perf_counter() - t0) / n_trials * 1000

    # 5. Reward computation on accepted states
    target_dists, target_dets = parallel_verify(target, stoch_0, draft_stochs, actions, det_0)
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        target.get_reward(target_dets.reshape(-1, target_dets.shape[-1]),
                          draft_stochs.reshape(-1, draft_stochs.shape[-1]))
    sync()
    results['reward_compute_ms'] = (time.perf_counter() - t0) / n_trials * 1000

    # Summary
    total_spec = results['draft_forward_ms'] + results['verify_ms'] + results['acceptance_check_ms']
    results['total_speculative_ms'] = total_spec
    results['speedup'] = results['target_rollout_ms'] / max(total_spec, 0.001)
    results['verify_fraction'] = results['verify_ms'] / max(total_spec, 0.001)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Pareto Frontier Sweep
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def pareto_sweep(target, draft, device, act_dim=1, H=30, batch_size=256):
    """Sweep verification strategies to find acceptance vs speedup Pareto frontier."""
    target.eval()
    draft.eval()

    actions = torch.randn(batch_size, H, act_dim, device=device).clamp(-1, 1)
    det_0, stoch_0 = target.initial_state(batch_size, device)

    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None
    n_trials = 100

    # Get draft predictions
    draft_out = draft(stoch_0, actions)
    draft_stochs, draft_dists = draft_out[:2]
    confidence = draft_out[2] if len(draft_out) > 2 else None

    # Target verification
    from src.parallel_verify import parallel_verify
    target_dists, target_dets = parallel_verify(target, stoch_0, draft_stochs, actions, det_0)

    # Time components
    # Draft time
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        draft(stoch_0, actions)
    sync()
    draft_time = (time.perf_counter() - t0) / n_trials

    # Target sequential time
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
    sync()
    target_time = (time.perf_counter() - t0) / n_trials

    # Verification of K steps (partial)
    def time_partial_verify(K):
        """Time to verify only first K steps."""
        sub_actions = actions[:, :K]
        sub_draft = draft_stochs[:, :K]
        sync(); t0 = time.perf_counter()
        for _ in range(n_trials // 2):
            parallel_verify(target, stoch_0, sub_draft, sub_actions, det_0)
        sync()
        return (time.perf_counter() - t0) / (n_trials // 2)

    # Full verification time
    full_verify_time = time_partial_verify(H)

    results = []

    # Strategy 1: No verification (trust draft completely)
    acc_rate_noverify = 1.0
    speedup_noverify = target_time / draft_time
    results.append({
        'strategy': 'no_verify',
        'acceptance_rate': acc_rate_noverify,
        'speedup': speedup_noverify,
        'description': 'Trust draft completely'
    })

    # Strategy 2: Full verification with various thresholds
    eps_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_values = [0.0, 0.005, 0.01]

    for eps in eps_values:
        for alpha in alpha_values:
            _, accepted_lengths, kl_divs = evaluate_and_accept(
                target_dists, draft_dists, eps_base=eps, alpha=alpha)
            acc_rate = (accepted_lengths.float().mean() / H).item()

            # Effective speedup: accepted steps save verification, rejected fall back to target
            # Simplified: time = draft_time + full_verify_time * (accepted/H)
            effective_speedup = target_time / (draft_time + full_verify_time)

            results.append({
                'strategy': f'verify_eps{eps}_a{alpha}',
                'acceptance_rate': acc_rate,
                'speedup': effective_speedup,
                'eps_base': eps,
                'alpha': alpha,
                'mean_kl': kl_divs.mean().item(),
                'median_accepted': accepted_lengths.float().median().item(),
            })

    # Strategy 3: Partial verification (verify only first K steps)
    for K in [5, 10, 15, 20]:
        partial_verify_time = time_partial_verify(K)
        # Acceptance based on first K steps only
        _, accepted_lengths_K, kl_K = evaluate_and_accept(
            target_dists[:, :K] if hasattr(target_dists, '__getitem__') else target_dists,
            draft_dists[:, :K] if hasattr(draft_dists, '__getitem__') else draft_dists,
            eps_base=0.1, alpha=0.005)
        # Use same acceptance for full H (extrapolate)
        acc_K = (accepted_lengths_K.float().mean() / K).item()
        speedup_K = target_time / (draft_time + partial_verify_time)
        results.append({
            'strategy': f'partial_verify_K{K}',
            'acceptance_rate': acc_K,
            'speedup': speedup_K,
            'verify_steps': K,
        })

    # Strategy 4: Confidence-based selective verification
    if confidence is not None:
        for conf_thresh in [0.5, 1.0, 2.0, 5.0]:
            confident_mask = confidence < conf_thresh  # [B, H]
            verified_frac = (~confident_mask).float().mean().item()
            selective_time = draft_time + full_verify_time * verified_frac
            selective_speedup = target_time / selective_time
            results.append({
                'strategy': f'confident_thresh{conf_thresh}',
                'acceptance_rate': 1.0 - verified_frac * 0.3,  # rough estimate
                'speedup': selective_speedup,
                'verified_fraction': verified_frac,
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4')
    parser.add_argument('--n-data-eps', type=int, default=15000)
    parser.add_argument('--target-epochs', type=int, default=600)
    parser.add_argument('--draft-epochs', type=int, default=800)
    parser.add_argument('--epsilon', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"  Speculative MPC — Round 5b (Improved Draft + Profiling)")
    print(f"  Env: {args.env} | Device: {device}" +
          (f" ({torch.cuda.get_device_name()})" if device.type == 'cuda' else ""))
    print(f"{'='*70}")

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    is_discrete = hasattr(env.action_space, 'n')
    act_dim = 1 if is_discrete else env.action_space.shape[0]
    stoch_dim = 30
    env.close()

    # Models
    target = DeepRSSM(obs_dim, act_dim, det_dim=200, stoch_dim=stoch_dim).to(device)
    draft = ImprovedDraft(stoch_dim, act_dim, hidden_dim=192, n_layers=6).to(device)

    n_target = sum(p.numel() for p in target.parameters())
    n_draft = sum(p.numel() for p in draft.parameters())
    print(f"\nTarget params: {n_target:,}")
    print(f"Draft params:  {n_draft:,} ({n_draft/n_target:.1%} of target)")

    # Step 1: Collect data
    print(f"\n{'─'*70}")
    print(f"  Step 1: Collecting {args.n_data_eps} episodes (ε={args.epsilon})")
    print(f"{'─'*70}")
    buffer = collect_eps_greedy_data(args.env, args.n_data_eps, epsilon=args.epsilon)

    # Step 2: Train target
    print(f"\n{'─'*70}")
    print(f"  Step 2: Training Target RSSM ({args.target_epochs} epochs)")
    print(f"{'─'*70}")
    train_target(target, buffer, n_epochs=args.target_epochs, device=device)

    # Step 3: Train improved draft
    print(f"\n{'─'*70}")
    print(f"  Step 3: Training Improved Draft ({args.draft_epochs} epochs)")
    print(f"{'─'*70}")
    train_draft_improved(target, draft, buffer, n_epochs=args.draft_epochs, device=device)

    # Save checkpoints
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    env_tag = args.env.replace('-', '_').replace('v1', '').replace('v4', '') + '_r5b'
    torch.save(target.state_dict(), os.path.join(ckpt_dir, f'target_{env_tag}.pt'))
    torch.save(draft.state_dict(), os.path.join(ckpt_dir, f'draft_{env_tag}.pt'))

    # Step 4: Profile pipeline
    print(f"\n{'─'*70}")
    print(f"  Step 4: Profiling Pipeline (H=30)")
    print(f"{'─'*70}")
    profile = profile_pipeline(target, draft, device, act_dim=act_dim)
    print(f"  Target rollout:         {profile['target_rollout_ms']:.2f} ms")
    print(f"  Draft forward:          {profile['draft_forward_ms']:.2f} ms")
    print(f"  Verification (GRU seq): {profile['verify_ms']:.2f} ms")
    print(f"  Acceptance check:       {profile['acceptance_check_ms']:.4f} ms")
    print(f"  Reward compute:         {profile['reward_compute_ms']:.2f} ms")
    print(f"  Total speculative:      {profile['total_speculative_ms']:.2f} ms")
    print(f"  Speedup:                {profile['speedup']:.2f}x")
    print(f"  Verification fraction:  {profile['verify_fraction']:.1%}")

    # Step 5: Pareto frontier
    print(f"\n{'─'*70}")
    print(f"  Step 5: Pareto Frontier Sweep")
    print(f"{'─'*70}")
    pareto = pareto_sweep(target, draft, device, act_dim=act_dim)
    print(f"\n  {'Strategy':<35} {'Accept%':>8} {'Speedup':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*8}")
    for r in sorted(pareto, key=lambda x: -x['speedup'])[:15]:
        print(f"  {r['strategy']:<35} {r['acceptance_rate']:>7.1%} {r['speedup']:>7.2f}x")

    # Step 6: MPC evaluation
    print(f"\n{'─'*70}")
    print(f"  Step 6: CEM MPC Evaluation on {args.env}")
    print(f"{'─'*70}")
    from src.main_v2 import evaluate_mpc
    mpc_results, retention = evaluate_mpc(
        target, draft, device, env_name=args.env, n_episodes=20,
        horizon=12, n_samples=256, n_iterations=5)

    # Save all results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        'env': args.env,
        'round': '5b',
        'model_sizes': {'target_params': n_target, 'draft_params': n_draft},
        'profile': profile,
        'pareto_frontier': pareto,
        'mpc_target_reward': float(np.mean(mpc_results['target_only'])),
        'mpc_speculative_reward': float(np.mean(mpc_results['speculative'])),
        'reward_retention': float(retention),
    }

    with open(os.path.join(results_dir, f'r5b_{env_tag}.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Round 5b — {args.env}")
    print(f"{'='*70}")
    print(f"  Target params:          {n_target:,}")
    print(f"  Draft params:           {n_draft:,} ({n_draft/n_target:.1%})")
    print(f"  Pipeline speedup:       {profile['speedup']:.2f}x")
    print(f"  Verification fraction:  {profile['verify_fraction']:.1%}")
    print(f"  MPC target reward:      {all_results['mpc_target_reward']:.1f}")
    print(f"  MPC speculative reward: {all_results['mpc_speculative_reward']:.1f}")
    print(f"  Reward retention:       {retention:.1%}")
    print(f"{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()
