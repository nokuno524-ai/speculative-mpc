"""H1: Verification cost is O(H) — measure time breakdown vs horizon."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.model_utils import load_target, load_draft, det_dim, stoch_dim, act_dim
from src.parallel_verify import parallel_verify

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    target = load_target(device)
    draft = load_draft(device)

    B = 512
    horizons = [5, 10, 20, 30, 50]
    n_warmup, n_trials = 5, 30

    results = {'draft': [], 'verify': [], 'target_only': []}

    for H in horizons:
        det_0 = torch.randn(B, det_dim, device=device)
        stoch_0 = torch.randn(B, stoch_dim, device=device)
        actions = torch.randn(B, H, act_dim, device=device).clamp(-1, 1)

        # 1. Draft forward pass
        for _ in range(n_warmup):
            draft(stoch_0, actions)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            draft(stoch_0, actions)
        if device.type == 'cuda': torch.cuda.synchronize()
        draft_time = (time.perf_counter() - t0) / n_trials

        # 2. Target-only rollout
        for _ in range(n_warmup):
            target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            target.unroll_imagine(det_0, stoch_0, actions, deterministic=True)
        if device.type == 'cuda': torch.cuda.synchronize()
        target_time = (time.perf_counter() - t0) / n_trials

        # 3. Verification: draft + parallel_verify
        for _ in range(n_warmup):
            ds, _ = draft(stoch_0, actions)
            parallel_verify(target, stoch_0, ds, actions, det_0)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            ds, _ = draft(stoch_0, actions)
            parallel_verify(target, stoch_0, ds, actions, det_0)
        if device.type == 'cuda': torch.cuda.synchronize()
        verify_time = (time.perf_counter() - t0) / n_trials

        results['draft'].append(draft_time * 1000)
        results['verify'].append(verify_time * 1000)
        results['target_only'].append(target_time * 1000)

        print(f"H={H:3d} | draft={draft_time*1000:.2f}ms | verify={verify_time*1000:.2f}ms | "
              f"target={target_time*1000:.2f}ms | verify/target={verify_time/target_time:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, results['draft'], 'o-', label='Draft (parallel)', linewidth=2)
    ax.plot(horizons, results['verify'], 's-', label='Draft + Verify (O(H) GRU)', linewidth=2)
    ax.plot(horizons, results['target_only'], '^-', label='Target-only (O(H))', linewidth=2)
    ax.set_xlabel('Horizon H')
    ax.set_ylabel('Time (ms)')
    ax.set_title('H1: Verification Cost vs Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiment_results')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'h1_verification_cost.png'), dpi=150)
    np.savez(os.path.join(out_dir, 'h1_data.npz'), horizons=horizons, **results)
    print(f"Saved to {out_dir}/")

if __name__ == '__main__':
    main()
