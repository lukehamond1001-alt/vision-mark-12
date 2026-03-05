"""Hierarchical encoder-decoder for Vision Mark 12.

4-level hierarchy:  char → word → phrase → idea
Encoder compresses bottom-up, decoder reconstructs top-down.
U-Net skip connections carry detail from encoder to decoder.

Level 0 reuses the existing Scanner + dilated blocks (VM12 baseline).
Levels 1-3 use Conv1d projection + dilated blocks on downsampled features.
Decoder mirrors the encoder with transposed convolutions for upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vm12.config import VM12Config, LevelConfig
from vm12.scanning_block import Scanner


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

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
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, bias=False)
        self.norm = RMSNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.pad, 0))
        h = self.conv(h)
        h = F.relu(self.norm(h))
        return x + h


class DownsampleBlock(nn.Module):
    """Stride-2 kernel-4 downsampling with causal padding."""

    def __init__(self, channels: int, kernel_size: int = 4, stride: int = 2):
        super().__init__()
        self.stride = stride
        # Causal: pad left so we don't see future positions
        self.pad = kernel_size - stride
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              stride=stride, bias=False)
        self.norm = RMSNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, seq_len) → (batch, channels, seq_len // stride)"""
        h = F.pad(x, (self.pad, 0))
        return F.relu(self.norm(self.conv(h)))


class UpsampleBlock(nn.Module):
    """Transposed convolution upsampling to restore positions."""

    def __init__(self, channels: int, kernel_size: int = 4, stride: int = 2):
        super().__init__()
        self.stride = stride
        # output_padding to match dimensions exactly
        self.deconv = nn.ConvTranspose1d(
            channels, channels, kernel_size, stride=stride,
            padding=(kernel_size - stride) // 2,
            output_padding=0, bias=False
        )
        self.norm = RMSNorm1d(channels)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """(batch, channels, seq_len) → (batch, channels, target_len)"""
        h = self.deconv(x)
        # Trim or pad to match encoder skip connection length
        if h.size(2) > target_len:
            h = h[:, :, :target_len]
        elif h.size(2) < target_len:
            h = F.pad(h, (0, target_len - h.size(2)))
        return F.relu(self.norm(h))


# ---------------------------------------------------------------------------
# Encoder levels
# ---------------------------------------------------------------------------

class EncoderLevel0(nn.Module):
    """Level 0: character-level scanner + dilated blocks.

    This is the original VM12 architecture, now as one level of the hierarchy.
    Input: (batch, seq_len) integer tokens
    Output: (batch, width, seq_len) features in channels-first format
    """

    def __init__(self, config: VM12Config):
        super().__init__()
        lc = config.level_configs[0]
        w = lc.width

        self.vocab_size = config.vocab_size
        self.scanner = Scanner(
            input_dim=config.vocab_size,
            max_fan_in=lc.max_fan_in,
            pairs_per_fan_in=lc.pairs_per_fan_in,
        )
        scanner_dim = self.scanner.values_per_unit

        self.project = nn.Conv1d(scanner_dim, w, 1, bias=False)
        self.blocks = nn.ModuleList([
            CausalDilatedBlock(w, lc.kernel_size, d)
            for d in lc.dilation_pattern
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) → (batch, width, seq_len)"""
        x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()
        scanned = self.scanner(x_onehot)           # (batch, seq_len, scanner_dim)
        h = scanned.permute(0, 2, 1)               # (batch, scanner_dim, seq_len)
        h = F.relu(self.project(h))                 # (batch, width, seq_len)
        for block in self.blocks:
            h = block(h)
        return h


class EncoderLevelN(nn.Module):
    """Level 1+: project downsampled features through dilated blocks.

    Input: (batch, width, seq_len) from downsampling the level below
    Output: (batch, width, seq_len) processed features
    """

    def __init__(self, level_config: LevelConfig, input_width: int):
        super().__init__()
        w = level_config.width

        # Project from previous level width to this level width
        self.project = nn.Conv1d(input_width, w, 1, bias=False) if input_width != w else nn.Identity()
        self.blocks = nn.ModuleList([
            CausalDilatedBlock(w, level_config.kernel_size, d)
            for d in level_config.dilation_pattern
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_width, seq_len) → (batch, width, seq_len)"""
        h = self.project(x)
        if not isinstance(self.project, nn.Identity):
            h = F.relu(h)
        for block in self.blocks:
            h = block(h)
        return h


# ---------------------------------------------------------------------------
# Decoder level
# ---------------------------------------------------------------------------

class DecoderLevel(nn.Module):
    """Single decoder level: upsample from above + skip from encoder.

    Takes the representation from the level above (upsampled to match spatial
    dimensions) and concatenates it with the encoder skip connection at this
    level, then processes through dilated blocks.
    """

    def __init__(self, level_config: LevelConfig, above_width: int):
        super().__init__()
        w = level_config.width

        # Upsample from the level above
        self.upsample = UpsampleBlock(
            above_width,
            kernel_size=level_config.downsample_kernel,
            stride=level_config.downsample_stride,
        )

        # Project upsampled (above_width) + skip (w) → w
        self.skip_proj = nn.Conv1d(above_width + w, w, 1, bias=False)

        self.blocks = nn.ModuleList([
            CausalDilatedBlock(w, level_config.kernel_size, d)
            for d in level_config.dilation_pattern
        ])

    def forward(self, x_above: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """Combine upsampled above + encoder skip, process through blocks.

        Args:
            x_above: (batch, above_width, above_seq_len) from decoder level above
            x_skip:  (batch, width, skip_seq_len) from encoder at this level
        Returns:
            (batch, width, skip_seq_len) decoded features
        """
        target_len = x_skip.size(2)
        x_up = self.upsample(x_above, target_len)   # (batch, above_width, target_len)
        x_cat = torch.cat([x_up, x_skip], dim=1)    # (batch, above_width + width, target_len)
        h = F.relu(self.skip_proj(x_cat))            # (batch, width, target_len)
        for block in self.blocks:
            h = block(h)
        return h


# ---------------------------------------------------------------------------
# Full hierarchical model
# ---------------------------------------------------------------------------

class HierarchicalModel(nn.Module):
    """Hierarchical Convolutional Language Model.

    Encoder: Level 0 (char) → downsample → Level 1 (word) → ... → Level 3 (idea)
    Decoder: Level 3 → upsample+skip → Level 2 → ... → Level 0 → output logits

    Dense skip connections: every encoder level feeds into the matching decoder
    level AND provides gradient paths to all levels below via the encoder chain.
    """

    def __init__(self, config: VM12Config):
        super().__init__()
        self.config = config
        n = config.num_levels

        # --- Encoder ---
        self.encoder_levels = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        # Level 0: character scanner
        self.encoder_levels.append(EncoderLevel0(config))

        # Levels 1+: project + dilated blocks
        for i in range(1, n):
            prev_lc = config.level_configs[i - 1]
            curr_lc = config.level_configs[i]

            # Downsample from previous level
            self.downsample_blocks.append(DownsampleBlock(
                prev_lc.width,
                kernel_size=prev_lc.downsample_kernel,
                stride=prev_lc.downsample_stride,
            ))

            # Encoder level
            self.encoder_levels.append(EncoderLevelN(curr_lc, prev_lc.width))

        # --- Decoder ---
        # Decoder levels go from level n-2 down to level 0
        # Decoder level i receives from decoder level i+1 (or encoder top) + encoder skip i
        self.decoder_levels = nn.ModuleList()
        for i in range(n - 2, -1, -1):
            above_idx = i + 1
            above_width = config.level_configs[above_idx].width
            self.decoder_levels.append(DecoderLevel(
                config.level_configs[i],
                above_width=above_width,
            ))

        # --- Dense skip connections ---
        # Project higher encoder levels down to level i's width and add
        # to encoder skip before passing to decoder.  This ensures every
        # encoder level receives gradient from ALL levels above it.
        self.dense_skips = nn.ModuleList()
        for i in range(n):
            level_projs = nn.ModuleList()
            for j in range(i + 1, n):
                level_projs.append(nn.Conv1d(
                    config.level_configs[j].width,
                    config.level_configs[i].width,
                    1, bias=False
                ))
            self.dense_skips.append(level_projs)

        # --- Output head ---
        self.output_head = nn.Conv1d(
            config.level_configs[0].width, config.vocab_size, 1, bias=False
        )

        # --- Contrastive projection heads (for levels 2 and 3) ---
        self.contrastive_heads = nn.ModuleDict()
        for i in [2, 3]:
            if i < n:
                self.contrastive_heads[str(i)] = nn.Sequential(
                    nn.Conv1d(config.level_configs[i].width, config.contrastive_dim, 1),
                    nn.ReLU(),
                    nn.Conv1d(config.contrastive_dim, config.contrastive_dim, 1),
                )

        # --- Word prediction head (for level 1) ---
        self.word_head = nn.Conv1d(
            config.level_configs[1].width if n > 1 else config.level_configs[0].width,
            config.level_configs[0].width,
            1, bias=False
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the encoder bottom-up.

        Args:
            x: (batch, seq_len) integer token ids

        Returns:
            encoder_features: list of (batch, width_i, seq_len_i) for each level
        """
        features = []

        # Level 0
        h = self.encoder_levels[0](x)  # (batch, w0, seq_len)
        features.append(h)

        # Levels 1+
        for i in range(1, self.config.num_levels):
            h = self.downsample_blocks[i - 1](h)    # (batch, w_{i-1}, seq_len/stride)
            h = self.encoder_levels[i](h)            # (batch, w_i, seq_len_i)
            features.append(h)

        return features

    def _enrich_skip(self, encoder_features: list[torch.Tensor],
                     level_idx: int) -> torch.Tensor:
        """Add dense skip projections from all higher encoder levels.

        For encoder level `level_idx`, project each higher-level feature
        (j > level_idx) down to level_idx's width and spatial resolution,
        then add to the skip connection. This ensures level_idx receives
        gradient from every level above, not just the one directly above.
        """
        skip = encoder_features[level_idx]  # (batch, w_i, seq_len_i)
        projs = self.dense_skips[level_idx]  # ModuleList of Conv1d projections

        for proj_idx, j in enumerate(range(level_idx + 1, self.config.num_levels)):
            higher = encoder_features[j]  # (batch, w_j, seq_len_j)
            projected = projs[proj_idx](higher)  # (batch, w_i, seq_len_j)

            # Upsample to match level_idx's spatial length
            target_len = skip.size(2)
            if projected.size(2) != target_len:
                projected = F.interpolate(
                    projected, size=target_len, mode='nearest'
                )

            skip = skip + projected

        return skip

    def decode(self, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """Run the decoder top-down with dense skip connections.

        At each decoder level, the encoder skip is enriched with projected
        features from ALL higher encoder levels before being concatenated
        with the upsampled decoder output from above.

        Args:
            encoder_features: list of encoder outputs, one per level

        Returns:
            (batch, width_0, seq_len_0) decoded features at character level
        """
        n = self.config.num_levels

        # Start from the top encoder level
        h = encoder_features[-1]

        # Decode downward: level n-2 → 0
        for dec_idx, level_idx in enumerate(range(n - 2, -1, -1)):
            skip = self._enrich_skip(encoder_features, level_idx)
            h = self.decoder_levels[dec_idx](h, skip)

        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass: encode → decode → logits.

        Args:
            x: (batch, seq_len) integer token ids

        Returns:
            dict with:
                'logits': (batch, seq_len, vocab_size) — char predictions
                'encoder_features': list of per-level encoder outputs
                'word_preds': (batch, w0, word_seq_len) — word-span predictions
                'contrastive_2': (batch, contrastive_dim, phrase_seq_len)
                'contrastive_3': (batch, contrastive_dim, idea_seq_len)
        """
        encoder_features = self.encode(x)
        decoded = self.decode(encoder_features)

        # Char-level output logits
        logits = self.output_head(decoded)            # (batch, vocab, seq_len)
        logits = logits.permute(0, 2, 1)              # (batch, seq_len, vocab)

        result = {
            'logits': logits,
            'encoder_features': encoder_features,
        }

        # Word-level predictions (if level 1 exists)
        if self.config.num_levels > 1:
            result['word_preds'] = self.word_head(encoder_features[1])

        # Contrastive embeddings for phrase and idea levels
        for level_str, head in self.contrastive_heads.items():
            level_idx = int(level_str)
            if level_idx < len(encoder_features):
                result[f'contrastive_{level_str}'] = head(encoder_features[level_idx])

        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_info(self) -> str:
        """Print a summary of the hierarchical architecture."""
        c = self.config
        lines = [
            "=== Hierarchical Convolutional LLM ===",
            f"Levels: {c.num_levels} | Vocab: {c.vocab_size}",
            "",
        ]

        total_rf = 0
        cumulative_stride = 1
        for i in range(c.num_levels):
            lc = c.level_configs[i]
            level_params = sum(
                p.numel() for p in self.encoder_levels[i].parameters()
            )

            # Receptive field for this level's dilated blocks
            block_rf = sum((lc.kernel_size - 1) * d for d in lc.dilation_pattern)
            if i == 0:
                scanner_rf = lc.max_fan_in
                level_rf = (scanner_rf + block_rf) * cumulative_stride
            else:
                level_rf = block_rf * cumulative_stride

            total_rf += level_rf
            lines.append(
                f"Encoder L{i}: {lc.width}ch, dilations={lc.dilation_pattern}, "
                f"RF={level_rf} chars — {level_params:,} params"
            )

            if i < c.num_levels - 1:
                cumulative_stride *= lc.downsample_stride
                ds_params = sum(
                    p.numel() for p in self.downsample_blocks[i].parameters()
                )
                lines.append(
                    f"  ↓ downsample: k={lc.downsample_kernel}, "
                    f"s={lc.downsample_stride} — {ds_params:,} params"
                )

        lines.append("")

        for dec_idx in range(len(self.decoder_levels)):
            dec_params = sum(
                p.numel() for p in self.decoder_levels[dec_idx].parameters()
            )
            level_idx = c.num_levels - 2 - dec_idx
            lines.append(f"Decoder L{level_idx}: {dec_params:,} params")

        lines.append("")
        lines.append(f"Total receptive field: ~{total_rf} characters")
        lines.append(f"Total parameters: {self.count_parameters():,}")

        return "\n".join(lines)
