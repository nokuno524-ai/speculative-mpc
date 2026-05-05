"""H2: Error compounds through the dynamics — per-step KL/MSE."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributions as td
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.model_utils import load_target, load_draft, det_dim, stoch_dim, act_dim

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    target = load_target(device)
    draft = load_draft(device)

    H, B, n_seeds = 50, 256, 10
    all_kl, all_mse = [], []

    with torch.no_grad():
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            det_0 = torch.randn(B, det_dim, device=device)
            stoch_0 = torch.randn(B, stoch_dim, device=device)
            actions = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

            # Target rollout
            det, stoch = det_0, stoch_0
            t_means, t_stds = [], []
            for t in range(H):
                det, stoch, prior = target.imagine_deterministic(det, stoch, actions[:, t])
                t_means.append(prior.mean)
                t_stds.append(prior.stddev)
            t_means = torch.stack(t_means, dim=1)
            t_stds = torch.stack(t_stds, dim=1)

            # Draft
            _, draft_dists = draft(stoch_0, actions)

            target_dists = td.Independent(td.Normal(t_means, t_stds), 1)
            kl = td.kl.kl_divergence(target_dists, draft_dists)
            mse = ((t_means - draft_dists.mean) ** 2).sum(-1)

            all_kl.append(kl.mean(0).cpu().numpy())
            all_mse.append(mse.mean(0).cpu().numpy())

    kl_curve = np.mean(all_kl, axis=0)
    mse_curve = np.mean(all_mse, axis=0)
    steps = np.arange(1, H + 1)

    log_kl = np.log(kl_curve + 1e-10)
    coeffs = np.polyfit(steps, log_kl, 1)
    growth_rate = coeffs[0]
    print(f"KL growth rate: {growth_rate:.4f}, step1={kl_curve[0]:.4f}, step{H}={kl_curve[-1]:.4f}, ratio={kl_curve[-1]/max(kl_curve[0],1e-10):.1f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, kl_curve, 'b-', linewidth=2)
    ax1.plot(steps, np.exp(np.polyval(coeffs, steps)), 'r--', label=f'Exp fit (slope={growth_rate:.3f})')
    ax1.set_xlabel('Timestep'); ax1.set_ylabel('KL'); ax1.set_title('H2: KL vs Timestep')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(steps, mse_curve, 'g-', linewidth=2)
    ax2.set_xlabel('Timestep'); ax2.set_ylabel('MSE'); ax2.set_title('H2: MSE vs Timestep')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'h2_error_compounding.png'), dpi=150)
    np.savez(os.path.join(out_dir, 'h2_data.npz'), steps=steps, kl_curve=kl_curve, mse_curve=mse_curve, growth_rate=growth_rate)
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
