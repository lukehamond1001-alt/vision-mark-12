"""Loss functions for hierarchical convolutional LLM.

Three loss types:
1. Character-level: standard next-char cross-entropy (Level 0)
2. Word-span: next-span-to-space MSE loss (Level 1)
3. Contrastive: InfoNCE for phrase/idea levels (Levels 2-3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


SPACE_TOKEN = 63  # from encode.py


class WordSpanLoss(nn.Module):
    """Next-span-to-space prediction loss for Level 1.

    At each space character position, predict the mean-pooled features
    of the next word span (characters until the following space).

    No external tokenizer needed — uses space (token 63) as word boundary.
    """

    def __init__(self, downsample_stride: int = 2):
        super().__init__()
        self.downsample_stride = downsample_stride

    def forward(self, word_preds: torch.Tensor, char_features: torch.Tensor,
                target_chars: torch.Tensor) -> torch.Tensor:
        """Compute word-span prediction loss.

        Args:
            word_preds:    (batch, feat_dim, word_seq_len) — predicted word vectors
            char_features: (batch, feat_dim, char_seq_len) — Level 0 encoder features
            target_chars:  (batch, char_seq_len) — target character ids

        Returns:
            Scalar MSE loss
        """
        batch_size = target_chars.size(0)
        total_loss = 0.0
        count = 0

        for b in range(batch_size):
            chars = target_chars[b]  # (char_seq_len,)
            feats = char_features[b]  # (feat_dim, char_seq_len)

            # Find space positions
            space_mask = (chars == SPACE_TOKEN)
            space_positions = space_mask.nonzero(as_tuple=True)[0]

            if len(space_positions) < 2:
                continue

            for idx in range(len(space_positions) - 1):
                pos = space_positions[idx].item()
                next_space = space_positions[idx + 1].item()

                if next_space - pos <= 1:
                    continue  # empty span

                # Target: mean-pooled char features of the next word span
                span_feats = feats[:, pos + 1:next_space]  # (feat_dim, span_len)
                target = span_feats.mean(dim=1)  # (feat_dim,)

                # Prediction: from the downsampled word-level at corresponding position
                word_pos = pos // self.downsample_stride
                if word_pos >= word_preds.size(2):
                    continue

                pred = word_preds[b, :, word_pos]  # (feat_dim,)
                total_loss += F.mse_loss(pred, target.detach())
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=target_chars.device, requires_grad=True)

        return total_loss / count


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for phrase and idea levels.

    Positive pairs: adjacent passages from the same document
    Negative pairs: passages from different documents in the batch

    No labels needed — pairs are sampled from position within the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss from positional contrastive pairs.

        Args:
            embeddings: (batch, contrastive_dim, seq_len) — projected features

        Returns:
            Scalar InfoNCE loss
        """
        batch_size, dim, seq_len = embeddings.shape

        if seq_len < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Create positive pairs: adjacent positions within each sequence
        # anchor: even positions, positive: odd positions (or next position)
        n_pairs = seq_len // 2

        # Gather anchors and positives
        anchors = embeddings[:, :, 0::2][:, :, :n_pairs]  # (batch, dim, n_pairs)
        positives = embeddings[:, :, 1::2][:, :, :n_pairs]  # (batch, dim, n_pairs)

        # Reshape: merge batch and pair dims
        # Each anchor-positive pair is one sample
        anchors = anchors.permute(0, 2, 1).reshape(-1, dim)  # (batch*n_pairs, dim)
        positives = positives.permute(0, 2, 1).reshape(-1, dim)  # (batch*n_pairs, dim)

        # L2 normalize
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)

        # Similarity matrix: each anchor against all positives
        n = anchors.size(0)
        logits = torch.mm(anchors, positives.t()) / self.temperature  # (n, n)

        # Labels: diagonal entries are the positive pairs
        labels = torch.arange(n, device=embeddings.device)

        return F.cross_entropy(logits, labels)


class HierarchicalLoss(nn.Module):
    """Combined loss for all levels of the hierarchy.

    Manages which losses are active based on training stage.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        stride = config.level_configs[0].downsample_stride
        self.word_loss = WordSpanLoss(downsample_stride=stride)
        self.contrastive_loss = InfoNCELoss(temperature=config.contrastive_temp)

    def forward(self, model_output: dict, targets: torch.Tensor,
                mask: torch.Tensor = None,
                active_stages: set = None) -> dict[str, torch.Tensor]:
        """Compute all active losses.

        Args:
            model_output: dict from HierarchicalModel.forward()
            targets: (batch, seq_len) target char ids
            mask: (batch, seq_len) optional loss mask (1=compute loss)
            active_stages: set of active stage names, e.g. {'char', 'word', 'contrastive'}

        Returns:
            dict of individual losses + 'total' combined loss
        """
        if active_stages is None:
            active_stages = {'char'}

        losses = {}

        # Level 0: Character-level cross-entropy
        if 'char' in active_stages:
            logits = model_output['logits']
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='none'
            ).reshape_as(targets)

            if mask is not None:
                n = mask.sum()
                if n > 0:
                    losses['char'] = (ce * mask).sum() / n
                else:
                    losses['char'] = ce.mean()
            else:
                losses['char'] = ce.mean()

        # Level 1: Word-span loss
        if 'word' in active_stages and 'word_preds' in model_output:
            losses['word'] = self.word_loss(
                model_output['word_preds'],
                model_output['encoder_features'][0],
                targets,
            )

        # Level 2: Phrase contrastive
        if 'contrastive' in active_stages and 'contrastive_2' in model_output:
            losses['phrase_contrastive'] = self.contrastive_loss(
                model_output['contrastive_2']
            )

        # Level 3: Idea contrastive
        if 'contrastive' in active_stages and 'contrastive_3' in model_output:
            losses['idea_contrastive'] = self.contrastive_loss(
                model_output['contrastive_3']
            )

        # Weighted total
        c = self.config
        total = torch.tensor(0.0, device=targets.device)
        if 'char' in losses:
            total = total + c.char_loss_weight * losses['char']
        if 'word' in losses:
            total = total + c.word_loss_weight * losses['word']
        if 'phrase_contrastive' in losses:
            total = total + c.phrase_loss_weight * losses['phrase_contrastive']
        if 'idea_contrastive' in losses:
            total = total + c.idea_loss_weight * losses['idea_contrastive']

        losses['total'] = total
        return losses
