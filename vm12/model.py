"""Vision Mark 12 — Convolutional LLM with parallel prediction.

All positions predict the next character simultaneously.
Multi-scale scanner feeds into dilated causal convolutions.
One forward pass → predictions at every position.
No attention, no embeddings, no autoregressive loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vm12.config import VM12Config
from vm12.scanning_block import Scanner


class RMSNorm1d(nn.Module):
    """RMS normalization for (batch, channels, seq_len) tensors."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight.view(1, -1, 1)


class CausalDilatedBlock(nn.Module):
    """Dilated causal Conv1d with RMSNorm, ReLU, and residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, bias=False)
        self.norm = RMSNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal padding on the left only
        h = F.pad(x, (self.pad, 0))
        h = self.conv(h)
        h = F.relu(self.norm(h))
        return x + h  # residual


class VM12Model(nn.Module):
    """Convolutional LLM — predicts next character at every position.

    Architecture:
        Input chars → one-hot(vocab_size)
        → Scanner: multi-scale Conv1d (kernels 1..K, P filters each)
        → Projection: Conv1d(scanner_dim → dense_width)
        → Dilated causal blocks (dilation 1, 2, 4, 8 — residual)
        → Output: Conv1d(dense_width → vocab_size) at ALL positions
    """

    def __init__(self, config: VM12Config):
        super().__init__()
        self.config = config
        w = config.dense_width

        # Multi-scale character scanner
        self.scanner = Scanner(
            input_dim=config.vocab_size,
            max_fan_in=config.word_max_fan_in,
            pairs_per_fan_in=config.pairs_per_fan_in,
        )
        scanner_dim = self.scanner.values_per_unit

        # Project scanner features to dense width
        self.project = nn.Conv1d(scanner_dim, w, 1, bias=False)

        # Dilated causal convolution stack
        self.blocks = nn.ModuleList([
            CausalDilatedBlock(w, config.dense_kernel_size, d)
            for d in config.dilation_pattern
        ])

        # Output head — predicts at every position
        self.output = nn.Conv1d(w, config.vocab_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) → (batch, seq_len, vocab_size)

        Every position i predicts the character at position i+1.
        """
        x_onehot = F.one_hot(x, num_classes=self.config.vocab_size).float()
        scanned = self.scanner(x_onehot)  # (batch, seq_len, scanner_dim)

        h = scanned.permute(0, 2, 1)  # (batch, scanner_dim, seq_len)
        h = F.relu(self.project(h))    # (batch, dense_width, seq_len)

        for block in self.blocks:
            h = block(h)               # (batch, dense_width, seq_len)

        logits = self.output(h)        # (batch, vocab_size, seq_len)
        return logits.permute(0, 2, 1) # (batch, seq_len, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_info(self) -> str:
        c = self.config
        scanner_dim = c.word_max_fan_in * c.pairs_per_fan_in
        w = c.dense_width
        dilations = c.dilation_pattern

        # Calculate effective receptive field
        scanner_rf = c.word_max_fan_in
        dense_rf = sum((c.dense_kernel_size - 1) * d for d in dilations)
        total_rf = scanner_rf + dense_rf

        scan_p = sum(p.numel() for p in self.scanner.parameters())
        proj_p = sum(p.numel() for p in self.project.parameters())
        block_p = sum(p.numel() for p in self.blocks.parameters())
        out_p = sum(p.numel() for p in self.output.parameters())

        lines = [
            f"=== Vision Mark 12 — Convolutional LLM ===",
            f"Vocab: {c.vocab_size} | All-position parallel prediction",
            f"",
            f"Scanner: one-hot({c.vocab_size}) → {scanner_dim}/pos "
            f"({c.pairs_per_fan_in} filters × {c.word_max_fan_in} scales) — {scan_p:,} params",
            f"Project: {scanner_dim} → {w} — {proj_p:,} params",
            f"Dilated blocks: {w}→{w} × {len(dilations)} "
            f"(dilations {dilations}, k={c.dense_kernel_size}) — {block_p:,} params",
            f"Output: Conv1d({w}, {c.vocab_size}, 1) — {out_p:,} params",
            f"",
            f"Receptive field: {total_rf} characters "
            f"(scanner={scanner_rf} + dense={dense_rf})",
            f"Total parameters: {self.count_parameters():,}",
        ]
        return "\n".join(lines)
