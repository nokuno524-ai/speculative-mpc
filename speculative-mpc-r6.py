"""R6: Complete the sweep — HalfCheetah + remaining CartPole configs.

r5 showed TransformerDraftV2 gets 100% acceptance on CartPole (up from 64.6%).
r5b collected 15k HalfCheetah episodes but died before training.
This script finishes both.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = '/scratch/qzp4ta/speculative-mpc'

# Reuse TransformerDraftV2 from r5
class TransformerDraftV2(nn.Module):
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


def evaluate_acceptance(target, draft, device, H=20, B=512):
    act_dim = draft.act_dim
    det_0 = torch.randn(B, 200, device=device) * 0.5
    s0 = torch.randn(B, 30, device=device) * 0.5
    acts = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

    with torch.no_grad():
        t_priors, t_dets, _ = target.unroll_imagine(det_0, s0, acts, deterministic=True)
        t_means = torch.stack([p.mean for p in t_priors], 1)
        t_stds = torch.stack([p.stddev for p in t_priors], 1)
        t_dists = td.Independent(td.Normal(t_means, t_stds), 1)

        _, d_means, d_dists = draft(det_0, s0, acts)

    kl = td.kl.kl_divergence(t_dists, d_dists)
    accept = (kl < 3.0).float()
    cum = torch.cumprod(accept, dim=1)
    rate = (cum.sum(1).mean() / H).item()
    full = (cum[:, -1] == 1).float().mean().item()
    return rate, full, kl.mean().item()


def profile_pipeline(target, draft, device, H=20, B=512):
    act_dim = draft.act_dim
    det_0 = torch.randn(B, 200, device=device) * 0.5
    s0 = torch.randn(B, 30, device=device) * 0.5
    acts = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            target.unroll_imagine(det_0, s0, acts, deterministic=True)
            draft(det_0, s0, acts)
    torch.cuda.synchronize()

    # Target timing
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            target.unroll_imagine(det_0, s0, acts, deterministic=True)
    torch.cuda.synchronize()
    target_ms = (time.perf_counter() - t0) / 50 * 1000

    # Draft timing
    t0 = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            draft(det_0, s0, acts)
    torch.cuda.synchronize()
    draft_ms = (time.perf_counter() - t0) / 50 * 1000

    speedup = target_ms / (draft_ms + 0.01)
    return {'target_rollout_ms': target_ms, 'draft_rollout_ms': draft_ms,
            'speedup': speedup}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['CartPole-v1', 'HalfCheetah-v4'])
    parser.add_argument('--train-steps', type=int, default=20000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env_tag = 'CartPole' if 'CartPole' in args.env else 'HalfCheetah'
    act_dim = 1 if 'CartPole' in args.env else 6

    print(f"R6: Complete Sweep | Device: {device} | Env: {args.env}")

    # Load target
    from src.main_v2 import DeepRSSM
    ckpt_path = f'{PROJECT}/checkpoints/target_{env_tag}_.pt'
    target = DeepRSSM(obs_dim=17 if env_tag=="HalfCheetah" else 4, act_dim=act_dim, det_dim=200, stoch_dim=30)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    target.load_state_dict(ckpt)
    target.to(device)
    target.eval()
    print(f"Target loaded: {env_tag}")

    # Configs to sweep
    if env_tag == 'CartPole':
        # Remaining from r5: configs 4-6
        configs = [
            {'n_layers': 4, 'tf_hidden': 512, 'n_steps': 10000},
            {'n_layers': 6, 'tf_hidden': 256, 'n_steps': 15000},
            {'n_layers': 4, 'tf_hidden': 256, 'n_steps': args.train_steps},
        ]
    else:
        # HalfCheetah: test a range
        configs = [
            {'n_layers': 2, 'tf_hidden': 256, 'n_steps': 10000},
            {'n_layers': 4, 'tf_hidden': 256, 'n_steps': 15000},
            {'n_layers': 4, 'tf_hidden': 512, 'n_steps': 15000},
            {'n_layers': 6, 'tf_hidden': 256, 'n_steps': 20000},
        ]

    results = []
    H_eval = 20
    best_acc = 0
    best_draft = None
    best_tag = None

    for i, cfg in enumerate(configs):
        tag = f"L{cfg['n_layers']}_D{cfg['tf_hidden']}_S{cfg['n_steps']}"
        print(f"\n{'='*60}\n  Config {i+1}/{len(configs)}: {tag}\n{'='*60}")

        draft = TransformerDraftV2(
            det_dim=200, stoch_dim=30, act_dim=act_dim,
            tf_hidden=cfg['tf_hidden'], n_layers=cfg['n_layers']).to(device)

        n_params = sum(p.numel() for p in draft.parameters())
        pct = n_params / sum(p.numel() for p in target.parameters()) * 100
        print(f"  Params: {n_params:,} ({pct:.1f}% of target)")

        train_draft(target, draft, device, n_steps=cfg['n_steps'],
                    B=64, H=H_eval, lr=1e-3)

        rate, full, mean_kl = evaluate_acceptance(target, draft, device, H=H_eval, B=512)
        prof = profile_pipeline(target, draft, device, H=H_eval, B=512)

        print(f"  Acceptance: {rate:.3f} ± full_chain={full:.3f} | "
              f"KL={mean_kl:.3f} | Speedup: {prof['speedup']:.2f}x")

        # Save checkpoint
        torch.save(draft.state_dict(), f'{PROJECT}/checkpoints/draft_r6_{env_tag}_{tag}.pt')

        r = {'tag': tag, 'n_params': n_params, 'acceptance': rate,
             'full_chain': full, 'mean_kl': mean_kl, **prof}
        results.append(r)

        if rate > best_acc:
            best_acc = rate
            best_draft = draft
            best_tag = tag
            print(f"  ★ New best: {rate:.3f}")

    # Save best
    torch.save(best_draft.state_dict(), f'{PROJECT}/checkpoints/draft_r6_{env_tag}_best.pt')

    # Horizon sweep with best
    print(f"\n{'='*60}\n  Horizon Sweep (best: {best_tag})\n{'='*60}")
    horizon_results = []
    for H in [5, 10, 20, 30, 50]:
        acc, full, kl = evaluate_acceptance(target, best_draft, device, H=H, B=512)
        prof = profile_pipeline(target, best_draft, device, H=H, B=512)
        horizon_results.append({'H': H, 'acceptance': acc, 'full_chain': full,
                                'mean_kl': kl, **prof})
        print(f"  H={H:3d} | acc={acc:.3f} | full={full:.3f} | speedup={prof['speedup']:.2f}x")

    # Save results
    out = {'env': args.env, 'configs': results, 'horizon_sweep': horizon_results,
           'best_config': best_tag}
    out_path = f'{PROJECT}/results/r6_sweep_{env_tag}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"  R6 SUMMARY — {args.env}")
    print(f"{'='*60}")
    for r in results:
        marker = '★' if r['tag'] == best_tag else ' '
        print(f" {marker} {r['tag']:30s} | acc={r['acceptance']:.3f} | "
              f"full={r.get('full_chain',0):.3f} | speedup={r['speedup']:.2f}x | params={r['n_params']:>8,}")
    print(f"\n  Best: {best_tag} → acceptance {best_acc:.3f}")
    print(f"{'='*60}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    tags = [r['tag'][:25] for r in results]
    axes[0].barh(tags, [r['acceptance'] for r in results], color='steelblue')
    axes[0].axvline(0.646, color='red', ls='--', label='h3 baseline (0.646)')
    axes[0].axvline(0.80, color='green', ls='--', label='Target 80%')
    axes[0].set_xlabel('Acceptance Rate')
    axes[0].set_title(f'Acceptance by Config ({args.env})')
    axes[0].legend(fontsize=7)

    axes[1].plot([h['H'] for h in horizon_results],
                 [h['acceptance'] for h in horizon_results], 'o-', label='Acceptance')
    axes[1].plot([h['H'] for h in horizon_results],
                 [h['full_chain'] for h in horizon_results], 's--', label='Full chain')
    axes[1].axhline(0.646, color='red', ls=':', alpha=0.5)
    axes[1].axhline(0.80, color='green', ls=':', alpha=0.5)
    axes[1].set_xlabel('Horizon H')
    axes[1].set_ylabel('Rate')
    axes[1].set_title('Acceptance vs Horizon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter([r['speedup'] for r in results],
                    [r['acceptance'] for r in results],
                    s=100, c='coral', edgecolors='black')
    for r in results:
        axes[2].annotate(r['tag'][:15], (r['speedup'], r['acceptance']),
                         fontsize=7, textcoords='offset points', xytext=(5, 5))
    axes[2].axhline(0.80, color='green', ls='--', alpha=0.5, label='80% target')
    axes[2].set_xlabel('Speedup')
    axes[2].set_ylabel('Acceptance')
    axes[2].set_title('Acceptance vs Speedup')
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f'{PROJECT}/experiment_results/r6_sweep_{env_tag}.png', dpi=150)
    print(f"Plot saved.")


if __name__ == '__main__':
    main()
