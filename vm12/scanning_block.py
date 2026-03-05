"""Multi-scale scanning layer for Vision Mark 12.

Each position gets examined by causal receptive fields of size 1..K.
P filters per kernel size detect different character-level patterns.
Output: fixed-length (batch, seq_len, K*P) — all scales at every position.
Fully differentiable, no WTA, no gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Scanner(nn.Module):
    """Multi-scale causal feature extractor.

    At each position, kernel_size=k examines k consecutive characters
    ending at that position (causal — no future leakage).
    P filters per kernel size, K kernel sizes → K*P features per position.
    """

    def __init__(self, input_dim: int, max_fan_in: int = 10,
                 pairs_per_fan_in: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.max_fan_in = max_fan_in
        self.pairs = pairs_per_fan_in
        self.values_per_unit = max_fan_in * pairs_per_fan_in

        # P filters per kernel size, implemented as Conv1d for speed
        self.convs = nn.ModuleList()
        for k in range(1, max_fan_in + 1):
            conv = nn.Conv1d(input_dim, pairs_per_fan_in, kernel_size=k,
                             padding=k - 1, bias=False)
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, input_dim) → (batch, seq_len, K*P)"""
        batch, seq_len, _ = x.shape

        h = x.permute(0, 2, 1)  # (batch, input_dim, seq_len) for Conv1d

        parts = []
        for k_idx in range(self.max_fan_in):
            # Causal conv: only look at current + past positions
            y = F.relu(self.convs[k_idx](h))[:, :, :seq_len]  # (batch, P, seq_len)
            parts.append(y)

        out = torch.cat(parts, dim=1)  # (batch, K*P, seq_len)
        return out.permute(0, 2, 1)  # (batch, seq_len, K*P)
