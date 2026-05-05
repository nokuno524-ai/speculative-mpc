"""H3: Transformer transition model — the make-or-break experiment."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.model_utils import (load_target, load_draft, RSSMTarget,
                                      obs_dim, act_dim, det_dim, stoch_dim, hidden_dim)


class TransformerTransition(nn.Module):
    """Transformer-based parallel world model transition."""
    def __init__(self, det_dim=200, stoch_dim=30, act_dim=1,
                 tf_hidden=256, n_heads=4, n_layers=2):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim

        self.act_embed = nn.Linear(act_dim, tf_hidden)
        self.init_proj = nn.Linear(stoch_dim + det_dim, tf_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=n_heads, dim_feedforward=tf_hidden*4,
            dropout=0.0, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.det_out = nn.Linear(tf_hidden, det_dim)
        self.prior_out = nn.Linear(tf_hidden, 2 * stoch_dim)

    def forward(self, det_0, stoch_0, actions):
        B, H, _ = actions.shape
        x = self.act_embed(actions)
        x = x + self.init_proj(torch.cat([stoch_0, det_0], -1)).unsqueeze(1)
        mask = nn.Transformer.generate_square_subsequent_mask(H, device=x.device)
        x = self.transformer(x, mask=mask)
        dets = self.det_out(x)
        params = self.prior_out(x)
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        dists = td.Independent(td.Normal(mean, std), 1)
        return dets, mean, dists


def train_transformer(target, trans, device, n_steps=5000, B=64, H=20):
    optimizer = torch.optim.Adam(trans.parameters(), lr=1e-3)
    target.eval()
    for step in range(n_steps):
        det_0 = torch.randn(B, det_dim, device=device)
        s0 = torch.randn(B, stoch_dim, device=device)
        acts = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

        with torch.no_grad():
            priors, t_dets, _ = target.unroll_imagine(det_0, s0, acts, deterministic=True)
            t_means = torch.stack([p.mean for p in priors], 1)
            t_stds = torch.stack([p.stddev for p in priors], 1)

        pred_dets, pred_stochs, pred_dists = trans(det_0, s0, acts)
        loss = F.mse_loss(pred_dets, t_dets) + F.mse_loss(pred_dists.mean, t_means)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(trans.parameters(), 1.0); optimizer.step()
        if step % 500 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    return trans


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    target = load_target(device)
    draft = load_draft(device)

    print("\n=== Training Transformer Transition ===")
    trans = TransformerTransition().to(device)
    trans = train_transformer(target, trans, device, n_steps=5000)
    trans.eval()

    print("\n=== Timing ===")
    horizons = [5, 10, 20, 30, 50]
    B = 512
    gru_t, trans_t = [], []
    for H in horizons:
        d0 = torch.randn(B, det_dim, device=device)
        s0 = torch.randn(B, stoch_dim, device=device)
        a = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)
        sync = lambda: torch.cuda.synchronize() if device.type == 'cuda' else None

        for _ in range(5): target.unroll_imagine(d0, s0, a, deterministic=True)
        sync(); t0 = time.perf_counter()
        for _ in range(30): target.unroll_imagine(d0, s0, a, deterministic=True)
        sync(); gt = (time.perf_counter() - t0) / 30

        for _ in range(5): trans(d0, s0, a)
        sync(); t0 = time.perf_counter()
        for _ in range(30): trans(d0, s0, a)
        sync(); tt = (time.perf_counter() - t0) / 30

        gru_t.append(gt * 1000); trans_t.append(tt * 1000)
        print(f"H={H:3d} | GRU={gt*1000:.2f}ms | Trans={tt*1000:.2f}ms | {gt/tt:.2f}x")

    # Speculative comparison
    print("\n=== Speculative Acceptance ===")
    H, B = 20, 256
    accept_rates = []
    for seed in range(10):
        torch.manual_seed(seed)
        d0 = torch.randn(B, det_dim, device=device)
        s0 = torch.randn(B, stoch_dim, device=device)
        a = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)
        with torch.no_grad():
            _, _, t_dists = trans(d0, s0, a)
            _, d_dists = draft(s0, a)
            kl = td.kl.kl_divergence(t_dists, d_dists)
            accept = (kl < 5.0).int()
            rate = torch.cumprod(accept, 1).sum(1).float().mean() / H
            accept_rates.append(rate.item())
    print(f"Transformer target acceptance: {np.mean(accept_rates):.3f} ± {np.std(accept_rates):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(horizons, gru_t, 'o-', label='GRU', linewidth=2)
    axes[0].plot(horizons, trans_t, 's-', label='Transformer', linewidth=2)
    axes[0].set_xlabel('H'); axes[0].set_ylabel('Time (ms)'); axes[0].set_title('H3: Speed')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(['Acceptance\nRate'], [np.mean(accept_rates)])
    axes[1].set_ylim(0, 1); axes[1].set_title('H3: Speculative Acceptance')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].text(0.5, 0.5, f'GRU→Trans speedup at H=50: {gru_t[-1]/trans_t[-1]:.1f}x\n'
                 f'Acceptance rate: {np.mean(accept_rates):.3f}',
                 ha='center', va='center', fontsize=14, transform=axes[2].transAxes)
    axes[2].set_title('H3: Summary')

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'h3_transformer_transition.png'), dpi=150)
    np.savez(os.path.join(out_dir, 'h3_data.npz'), horizons=horizons, gru_times=gru_t,
             trans_times=trans_t, accept_rates=accept_rates)
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
