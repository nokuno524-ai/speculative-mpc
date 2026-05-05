"""Non-autoregressive draft model: predicts all H states in one forward pass."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class MLDraft(nn.Module):
    """Non-autoregressive draft: single MLP predicts all H stochastic states.

    Takes (stoch_0, actions[0:H]) → predicts all H stochastic states at once.
    O(1) wall-clock vs O(H) for autoregressive target.
    """

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
        """Predict all H stochastic states in one pass.

        Args:
            stoch_0: [B, stoch_dim]
            actions: [B, H, act_dim]
        Returns:
            pred_stochs: [B, H, stoch_dim] — predicted states (mean)
            dists: Independent Normal [B, H, stoch_dim]
        """
        B, H, _ = actions.shape

        # Expand stoch_0 to concat with each action
        s0 = stoch_0.unsqueeze(1).expand(-1, H, -1)  # [B, H, stoch_dim]
        inp = torch.cat([s0, actions], dim=-1)  # [B, H, stoch_dim + act_dim]

        flat = inp.reshape(B * H, -1)
        params = self.net(flat).reshape(B, H, -1)  # [B, H, 2*stoch_dim]

        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1

        dists = td.Independent(td.Normal(mean, std), 1)
        pred_stochs = mean  # deterministic for drafting

        return pred_stochs, dists
