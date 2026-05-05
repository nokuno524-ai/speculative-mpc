"""R5: Improve distilled transformer draft → push acceptance >80%.

Key improvements over h3:
  1. Better training: longer, cosine LR, diverse data
  2. Architecture: Pre-LN transformer, configurable depth/width, residual init projection
  3. Loss: MSE + cosine + distributional KL on std params
  4. Sweep configs → Pareto frontier
  5. Profile pipeline stages
  6. Test on HalfCheetah with existing target checkpoint
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TransformerDraftV2(nn.Module):
    """Improved transformer transition for speculative verification."""
    def __init__(self, det_dim=200, stoch_dim=30, act_dim=1,
                 tf_hidden=256, n_heads=4, n_layers=4):
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


def train_draft(target, draft, device, n_steps=10000, B=64, H=20, lr=1e-3):
    optimizer = torch.optim.AdamW(draft.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=lr * 0.01)
    target.eval()
    act_dim = draft.act_dim

    for step in range(n_steps):
        det_0 = torch.randn(B, target.det_dim, device=device) * 0.5
        s0 = torch.randn(B, target.stoch_dim, device=device) * 0.5
        acts = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

        with torch.no_grad():
            priors, t_dets, _ = target.unroll_imagine(det_0, s0, acts, deterministic=True)
            t_means = torch.stack([p.mean for p in priors], 1)
            t_stds = torch.stack([p.stddev for p in priors], 1)

        pred_dets, pred_means, pred_dists = draft(det_0, s0, acts)

        mean_loss = F.mse_loss(pred_means, t_means)
        det_loss = F.mse_loss(pred_dets, t_dets)
        cos_loss = 1.0 - F.cosine_similarity(pred_means, t_means, dim=-1).mean()
        # Std matching loss
        std_loss = F.mse_loss(pred_dists.stddev, t_stds)

        loss = det_loss + mean_loss + 0.5 * cos_loss + 0.2 * std_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 2000 == 0:
            print(f"  Step {step}/{n_steps} | loss={loss.item():.4f} "
                  f"(mean={mean_loss.item():.4f}, det={det_loss.item():.4f}, "
                  f"cos={cos_loss.item():.4f}, std={std_loss.item():.4f})")

    return draft


@torch.no_grad()
def evaluate_acceptance(target, draft, device, H=20, B=512, n_trials=20, eps_base=5.0):
    target.eval(); draft.eval()
    rates, kl_per_step = [], []
    act_dim = draft.act_dim

    for _ in range(n_trials):
        det_0 = torch.randn(B, target.det_dim, device=device) * 0.5
        s0 = torch.randn(B, target.stoch_dim, device=device) * 0.5
        acts = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

        priors, _, _ = target.unroll_imagine(det_0, s0, acts, deterministic=True)
        t_means = torch.stack([p.mean for p in priors], 1)
        t_stds = torch.stack([p.stddev for p in priors], 1)
        t_dists = td.Independent(td.Normal(t_means, t_stds), 1)

        _, _, d_dists = draft(det_0, s0, acts)

        kl = td.kl.kl_divergence(t_dists, d_dists)
        kl_per_step.append(kl.mean(0).cpu().numpy())

        accept = (kl < eps_base).float()
        cum_accept = torch.cumprod(accept, dim=1)
        rates.append((cum_accept.sum(1).mean() / H).item())

    return np.mean(rates), np.std(rates), np.mean(np.stack(kl_per_step), axis=0)


@torch.no_grad()
def profile_pipeline(target, draft, device, H=20, B=512, n_trials=50):
    target.eval(); draft.eval()
    act_dim = draft.act_dim
    det_0 = torch.randn(B, target.det_dim, device=device)
    s0 = torch.randn(B, target.stoch_dim, device=device)
    acts = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)
    sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None

    for _ in range(10):
        target.unroll_imagine(det_0, s0, acts, deterministic=True)
        draft(det_0, s0, acts)

    # Target rollout
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        priors, dets, stochs = target.unroll_imagine(det_0, s0, acts, deterministic=True)
        rewards = target.get_reward(dets.reshape(-1, dets.shape[-1]),
                                     stochs.reshape(-1, stochs.shape[-1]))
    sync(); target_time = (time.perf_counter() - t0) / n_trials

    # Draft forward
    sync(); t0 = time.perf_counter()
    for _ in range(n_trials):
        dets, means, dists = draft(det_0, s0, acts)
        rewards = target.get_reward(dets.reshape(-1, dets.shape[-1]),
                                     means.reshape(-1, means.shape[-1]))
    sync(); draft_reward_time = (time.perf_counter() - t0) / n_trials

    return {
        'target_rollout_ms': target_time * 1000,
        'draft_plus_reward_ms': draft_reward_time * 1000,
        'speedup': target_time / draft_reward_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'HalfCheetah-v4'])
    parser.add_argument('--train-steps', type=int, default=15000)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"R5: Improve Transformer Draft | Device: {device} | Env: {args.env}")

    PROJECT = '/scratch/qzp4ta/speculative-mpc'
    sys.path.insert(0, PROJECT)
    from experiments.model_utils import RSSMTarget

    env_tag = 'CartPole' if 'CartPole' in args.env else 'HalfCheetah'
    act_dim = 1 if 'CartPole' in args.env else 6
    obs_dim_env = 4 if 'CartPole' in args.env else 17

    target = RSSMTarget(obs_dim=obs_dim_env, act_dim=act_dim).to(device)
    ckpt = torch.load(f'{PROJECT}/checkpoints/target_{env_tag}_.pt',
                       map_location=device, weights_only=True)
    target.load_state_dict(ckpt)
    target.eval()
    print(f"Target loaded: {env_tag}")

    if args.quick:
        configs = [
            {'n_layers': 4, 'tf_hidden': 256, 'n_steps': 3000},
            {'n_layers': 6, 'tf_hidden': 256, 'n_steps': 5000},
        ]
    else:
        configs = [
            {'n_layers': 2, 'tf_hidden': 256, 'n_steps': 5000},   # h3 baseline equiv
            {'n_layers': 4, 'tf_hidden': 256, 'n_steps': 10000},
            {'n_layers': 2, 'tf_hidden': 512, 'n_steps': 10000},
            {'n_layers': 4, 'tf_hidden': 512, 'n_steps': 10000},
            {'n_layers': 6, 'tf_hidden': 256, 'n_steps': 15000},
            {'n_layers': 4, 'tf_hidden': 256, 'n_steps': args.train_steps},
        ]

    results = []
    H_eval = 20

    for i, cfg in enumerate(configs):
        tag = f"L{cfg['n_layers']}_D{cfg['tf_hidden']}_S{cfg['n_steps']}"
        print(f"\n{'='*60}\n  Config {i+1}/{len(configs)}: {tag}\n{'='*60}")

        draft = TransformerDraftV2(
            det_dim=200, stoch_dim=30, act_dim=act_dim,
            tf_hidden=cfg['tf_hidden'], n_layers=cfg['n_layers'],
            n_heads=max(4, cfg['tf_hidden'] // 64),
        ).to(device)

        n_params = sum(p.numel() for p in draft.parameters())
        n_target = sum(p.numel() for p in target.parameters())
        print(f"  Params: {n_params:,} ({n_params/n_target:.1%} of target)")

        draft = train_draft(target, draft, device, n_steps=cfg['n_steps'],
                            B=64, H=H_eval, lr=1e-3)

        acc_mean, acc_std, kl_per_step = evaluate_acceptance(
            target, draft, device, H=H_eval, B=512, n_trials=20)
        prof = profile_pipeline(target, draft, device, H=H_eval, B=512)
        print(f"  Acceptance: {acc_mean:.3f} ± {acc_std:.3f} | "
              f"Speedup: {prof['speedup']:.2f}x")

        results.append({
            'tag': tag, 'n_layers': cfg['n_layers'],
            'tf_hidden': cfg['tf_hidden'], 'train_steps': cfg['n_steps'],
            'n_params': n_params, 'acceptance': acc_mean,
            'acceptance_std': acc_std, 'kl_per_step': kl_per_step.tolist(),
            **prof,
        })

        if acc_mean > max((r['acceptance'] for r in results[:-1]), default=0):
            torch.save(draft.state_dict(),
                       f'{PROJECT}/checkpoints/draft_transformer_{env_tag}_best.pt')
            print(f"  ★ New best: {acc_mean:.3f}")

    # Horizon sweep with best
    best = max(results, key=lambda r: r['acceptance'])
    print(f"\n{'='*60}\n  Horizon Sweep | Best: {best['tag']}\n{'='*60}")

    best_draft = TransformerDraftV2(
        det_dim=200, stoch_dim=30, act_dim=act_dim,
        tf_hidden=best['tf_hidden'], n_layers=best['n_layers'],
        n_heads=max(4, best['tf_hidden'] // 64),
    ).to(device)
    best_draft.load_state_dict(
        torch.load(f'{PROJECT}/checkpoints/draft_transformer_{env_tag}_best.pt',
                    map_location=device, weights_only=True))
    best_draft.eval()

    horizon_results = []
    for H in [5, 10, 20, 30, 50]:
        acc, _, _ = evaluate_acceptance(target, best_draft, device, H=H, B=512)
        prof = profile_pipeline(target, best_draft, device, H=H, B=512)
        horizon_results.append({'H': H, 'acceptance': acc, **prof})
        print(f"  H={H:3d} | acc={acc:.3f} | target={prof['target_rollout_ms']:.2f}ms | "
              f"draft+rew={prof['draft_plus_reward_ms']:.2f}ms | speedup={prof['speedup']:.2f}x")

    # KL threshold Pareto
    print(f"\n{'='*60}\n  KL Threshold Pareto (H=20)\n{'='*60}")
    B_p = 1024
    det_0 = torch.randn(B_p, 200, device=device) * 0.5
    s0 = torch.randn(B_p, 30, device=device) * 0.5
    acts = torch.randn(B_p, 20, act_dim, device=device).clamp(-1, 1)
    with torch.no_grad():
        priors, _, _ = target.unroll_imagine(det_0, s0, acts, deterministic=True)
        t_means = torch.stack([p.mean for p in priors], 1)
        t_stds = torch.stack([p.stddev for p in priors], 1)
        t_dists = td.Independent(td.Normal(t_means, t_stds), 1)
        _, _, d_dists = best_draft(det_0, s0, acts)
        kl = td.kl.kl_divergence(t_dists, d_dists)

    pareto = []
    for eps in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        accept = (kl < eps).float()
        cum = torch.cumprod(accept, dim=1)
        rate = (cum.sum(1).mean() / 20).item()
        full = (cum[:, -1] == 1).float().mean().item()
        pareto.append({'eps': eps, 'rate': rate, 'full': full})
        print(f"  eps={eps:5.1f} | rate={rate:.3f} | full={full:.3f}")

    # Save
    out = {'env': args.env, 'configs': results, 'horizon_sweep': horizon_results,
           'kl_pareto': pareto, 'best_config': best['tag']}
    out_path = f'{PROJECT}/results/r5_improved_draft_{env_tag}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"  R5 SUMMARY — {args.env}")
    print(f"{'='*60}")
    for r in results:
        marker = '★' if r['tag'] == best['tag'] else ' '
        print(f" {marker} {r['tag']:30s} | acc={r['acceptance']:.3f} | "
              f"speedup={r['speedup']:.2f}x | params={r['n_params']:>8,}")
    print(f"\n  Best: {best['tag']} → acceptance {best['acceptance']:.3f} "
          f"(h3 baseline: 0.646)")
    print(f"{'='*60}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tags = [r['tag'][:25] for r in results]
    axes[0, 0].barh(tags, [r['acceptance'] for r in results], color='steelblue')
    axes[0, 0].axvline(0.646, color='red', ls='--', label='h3 baseline (0.646)')
    axes[0, 0].axvline(0.80, color='green', ls='--', label='Target 80%')
    axes[0, 0].set_xlabel('Acceptance Rate')
    axes[0, 0].set_title(f'Acceptance by Config ({args.env})')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot([p['rate'] for p in pareto], [p['eps'] for p in pareto],
                     'o-', color='darkgreen')
    axes[0, 1].set_xlabel('Acceptance Rate'); axes[0, 1].set_ylabel('KL Threshold')
    axes[0, 1].set_title('Acceptance vs KL Threshold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot([h['H'] for h in horizon_results],
                     [h['target_rollout_ms'] for h in horizon_results],
                     'o-', label='Target (GRU)')
    axes[1, 0].plot([h['H'] for h in horizon_results],
                     [h['draft_plus_reward_ms'] for h in horizon_results],
                     's-', label='Draft+Reward (Trans)')
    axes[1, 0].set_xlabel('Horizon H'); axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].set_title('Timing vs Horizon')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter([r['speedup'] for r in results],
                        [r['acceptance'] for r in results],
                        s=100, c='coral', edgecolors='black')
    for r in results:
        axes[1, 1].annotate(r['tag'][:15], (r['speedup'], r['acceptance']),
                             fontsize=7, textcoords='offset points', xytext=(5, 5))
    axes[1, 1].axhline(0.646, color='red', ls='--', alpha=0.5, label='h3 baseline')
    axes[1, 1].axhline(0.80, color='green', ls='--', alpha=0.5, label='Target 80%')
    axes[1, 1].set_xlabel('Speedup'); axes[1, 1].set_ylabel('Acceptance')
    axes[1, 1].set_title('Acceptance vs Speedup')
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{PROJECT}/experiment_results/r5_improved_draft_{env_tag}.png', dpi=150)
    print(f"Plot saved.")


if __name__ == '__main__':
    main()
