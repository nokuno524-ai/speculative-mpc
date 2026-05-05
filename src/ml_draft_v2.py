"""Non-autoregressive draft model v2: uses causal convolutions for temporal context.

v1 problem: each timestep only saw stoch_0 + action_t (no temporal context).
v2 solution: 1D causal convolution over the action sequence provides context.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class CausalConvDraft(nn.Module):
    """Draft model using causal 1D convolutions over action sequences.

    Architecture:
    1. Embed actions + positional encoding
    2. Causal conv layers capture temporal dependencies
    3. Cross-attend with stoch_0 to condition on initial state
    4. Output all H stochastic states in one forward pass

    Still O(1) wall-clock (parallel conv), but much better temporal modeling.
    """

    def __init__(self, stoch_dim, act_dim, hidden_dim=128, n_layers=4, kernel_size=3):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Action embedding
        self.act_embed = nn.Linear(act_dim, hidden_dim)

        # Positional encoding (learned, up to 100 steps)
        self.pos_embed = nn.Embedding(100, hidden_dim)

        # Causal convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                          padding=kernel_size - 1,  # causal padding
                          groups=1),
                nn.GELU(),
            ))

        # Initial state projection
        self.state_proj = nn.Linear(stoch_dim, hidden_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def forward(self, stoch_0, actions):
        """Predict all H stochastic states in one pass.

        Args:
            stoch_0: [B, stoch_dim]
            actions: [B, H, act_dim]
        Returns:
            pred_stochs: [B, H, stoch_dim]
            dists: Independent Normal [B, H, stoch_dim]
        """
        B, H, _ = actions.shape

        # Embed actions + positions
        act_emb = self.act_embed(actions)  # [B, H, hidden]
        positions = torch.arange(H, device=actions.device)
        pos_emb = self.pos_embed(positions)  # [H, hidden]
        x = act_emb + pos_emb  # [B, H, hidden]

        # Add stoch_0 conditioning (broadcast)
        state_emb = self.state_proj(stoch_0)  # [B, hidden]
        x = x + state_emb.unsqueeze(1)  # [B, H, hidden]

        # Causal convolutions (need [B, C, T] format)
        x = x.transpose(1, 2)  # [B, hidden, H]
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            # Trim to causal (remove future padding)
            x = x[:, :, :H]
            x = x + residual  # residual connection
        x = x.transpose(1, 2)  # [B, H, hidden]

        # Output
        params = self.output_head(x)  # [B, H, 2*stoch_dim]
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1

        dists = td.Independent(td.Normal(mean, std), 1)
        pred_stochs = mean

        return pred_stochs, dists


# Keep original for backward compat
class MLDraft(nn.Module):
    """Original non-autoregressive draft (v1, kept for comparison)."""
    def __init__(self, stoch_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.act_dim = act_dim

        self.net = nn.Sequential(
            nn.Linear(stoch_dim + act_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def forward(self, stoch_0, actions):
        B, H, _ = actions.shape
        s0 = stoch_0.unsqueeze(1).expand(-1, H, -1)
        inp = torch.cat([s0, actions], dim=-1)
        flat = inp.reshape(B * H, -1)
        params = self.net(flat).reshape(B, H, -1)
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        dists = td.Independent(td.Normal(mean, std), 1)
        pred_stochs = mean
        return pred_stochs, dists
