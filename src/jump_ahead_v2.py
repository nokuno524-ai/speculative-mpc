"""Improved jump-ahead training with normalization and proper loss weighting."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ImprovedJumpAheadModel(nn.Module):
    """Jump-ahead with better architecture: LayerNorm, residual connections."""

    def __init__(self, stoch_dim, act_dim, jump_k=5, hidden_dim=256):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.jump_k = jump_k

        # State encoder
        self.state_norm = nn.LayerNorm(stoch_dim)

        # Action sequence encoder with small transformer-like attention
        self.act_embed = nn.Linear(act_dim, hidden_dim)
        self.act_pos = nn.Parameter(torch.randn(1, jump_k, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=0.1, activation='gelu', batch_first=True)
        self.act_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.act_pool = nn.Linear(hidden_dim * jump_k, hidden_dim)

        # State transition predictor with residual
        self.predictor = nn.Sequential(
            nn.Linear(stoch_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(stoch_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, jump_k),
        )
        self.act_pool2 = nn.Linear(hidden_dim * jump_k, hidden_dim)

    def forward(self, stoch_t, actions_chunk):
        B = stoch_t.shape[0]
        k = self.jump_k

        # Encode action sequence
        act_emb = self.act_embed(actions_chunk)  # [B, k, hidden]
        act_emb = act_emb + self.act_pos[:, :k]
        act_encoded = self.act_encoder(act_emb)  # [B, k, hidden]
        act_flat = self.act_pool(act_encoded.reshape(B, -1))  # [B, hidden]

        # Normalize input state
        stoch_normed = self.state_norm(stoch_t)

        # Predict state delta (residual learning)
        combined = torch.cat([stoch_normed, act_flat], dim=-1)
        delta = self.predictor(combined)
        stoch_next = stoch_t + delta  # residual

        # Reward prediction
        act_flat2 = self.act_pool2(act_encoded.reshape(B, -1))
        rewards = self.reward_head(torch.cat([stoch_normed, act_flat2], dim=-1))

        return stoch_next, rewards


def train_jump_ahead_improved(target, jump_model, buffer, n_epochs=800,
                               batch_size=256, seq_len=60, lr=1e-4, device='cpu'):
    """Train with normalized targets, state-only pretraining, cosine loss."""
    optimizer = optim.AdamW(jump_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    target.eval()
    jump_model.train()
    k = jump_model.jump_k

    # Compute state statistics for normalization
    print("  Computing state statistics...")
    all_stochs = []
    with torch.no_grad():
        for _ in range(50):
            obs, acts, _ = buffer.sample_sequences(256, 30)
            obs, acts = obs.to(device), acts.to(device)
            B = obs.shape[0]
            det, stoch = target.initial_state(B, device)
            for t in range(min(30, obs.shape[1])):
                det, stoch, _ = target.observe(obs[:, t], det, stoch, acts[:, t])
                all_stochs.append(stoch.cpu())
    all_stochs = torch.cat(all_stochs, dim=0)
    stoch_mean = all_stochs.mean(0).to(device)
    stoch_std = all_stochs.std(0).clamp(min=1e-4).to(device)
    print(f"  State mean range: [{stoch_mean.min():.4f}, {stoch_mean.max():.4f}]")
    print(f"  State std range: [{stoch_std.min():.4f}, {stoch_std.max():.4f}]")

    best_loss = float('inf')
    patience = 0

    for epoch in range(n_epochs):
        obs, acts, rews = buffer.sample_sequences(batch_size, seq_len)
        obs, acts, rews = obs.to(device), acts.to(device), rews.to(device)
        B, T = obs.shape[:2]

        # Get target states
        with torch.no_grad():
            det, stoch = target.initial_state(B, device)
            target_stochs = []
            for t in range(T):
                det, stoch, _ = target.observe(obs[:, t], det, stoch, acts[:, t])
                target_stochs.append(stoch)
            target_stochs = torch.stack(target_stochs, dim=1)

        max_start = T - k
        if max_start < 1:
            continue

        # Multiple training pairs per batch for efficiency
        n_pairs = 3
        total_loss = 0
        for _ in range(n_pairs):
            starts = torch.randint(0, max_start, (B,), device=device)
            stoch_t = torch.stack([target_stochs[b, s] for b, s in enumerate(starts)])
            stoch_tp1 = torch.stack([target_stochs[b, s + k] for b, s in enumerate(starts)])
            act_chunks = torch.stack([acts[b, s:s+k] for b, s in enumerate(starts)])
            rew_chunks = torch.stack([rews[b, s:s+k] for b, s in enumerate(starts)])

            optimizer.zero_grad()
            pred_stoch, pred_rews = jump_model(stoch_t, act_chunks)

            # Normalized MSE for state
            pred_norm = (pred_stoch - stoch_mean) / stoch_std
            tgt_norm = (stoch_tp1 - stoch_mean) / stoch_std
            state_loss = F.mse_loss(pred_norm, tgt_norm)

            # Cosine similarity loss (encourage direction matching)
            cos_loss = (1 - F.cosine_similarity(pred_stoch, stoch_tp1, dim=-1)).mean()

            # Reward loss (normalized)
            rew_loss = F.mse_loss(pred_rews, rew_chunks)

            # State prediction is the primary objective
            loss = 10.0 * state_loss + 5.0 * cos_loss + 0.1 * rew_loss

            loss.backward()
            total_loss += loss.item()

        nn.utils.clip_grad_norm_(jump_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                mse = F.mse_loss(pred_stoch, stoch_tp1).item()
                cos = F.cosine_similarity(pred_stoch, stoch_tp1, dim=-1).mean().item()
            print(f"  Jump epoch {epoch+1}/{n_epochs} | "
                  f"MSE: {mse:.6f} | Cos: {cos:.4f} | Rew: {rew_loss.item():.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

            if total_loss < best_loss:
                best_loss = total_loss
                patience = 0
            else:
                patience += 1
                if patience >= 5:  # 500 epochs no improvement
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    return jump_model
