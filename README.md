# Vision Mark 12 — Hierarchical Convolutional LLM

Convolutional language model that compresses characters into word → phrase → idea representations, then decodes back to text.

**31.9M parameters** · **~460 char receptive field** · **No attention, no embeddings**

## Architecture

```
INPUT CHARACTERS (batch, seq_len)
      ↓ one-hot(99)
Level 0: Scanner (k=1..10) + Dilated Blocks [1,2,4,8]     ← 3.2M params
      ↓ downsample (Conv1d k=4, stride=2)
Level 1: Word Scanner + Dilated Blocks [1,2,4,8]           ← 3.1M params
      ↓ downsample (Conv1d k=4, stride=2)
Level 2: Phrase Scanner + Dilated Blocks [1,2,4,8]         ← 3.1M params
      ↓ downsample (Conv1d k=4, stride=2)
Level 3: Idea Scanner + Dilated Blocks [1,2,4,8]           ← 3.1M params
      ↓
      IDEA REPRESENTATION
      ↓
Decoder L2: Upsample + Skip(Encoder L2) + Dilated Blocks   ← 4.7M params
Decoder L1: Upsample + Skip(Encoder L1) + Dilated Blocks   ← 4.7M params
Decoder L0: Upsample + Skip(Encoder L0) + Dilated Blocks   ← 4.7M params
      ↓
OUTPUT LOGITS (batch, seq_len, 99)
```

## What's New (v12.1 — Hierarchical)

The original VM12 had a ~40 character receptive field, which was too small for coherent generation. The hierarchical version:

- **4-level encoder** stacks scanner + dilated blocks, downsampling by 2× between levels
- **U-Net decoder** with skip connections from encoder → decoder at each level
- **Dense skip connections** so every level gets gradient from all levels above
- **3 loss types**: char prediction (L0), word-span-to-space (L1), contrastive InfoNCE (L2-L3)
- **6-stage training**: freeze lower levels as you build upward, then joint fine-tune

### Receptive Field

| Level | Effective span |
|-------|---------------|
| 0 — Char | ~40 chars |
| 1 — Word | ~60 chars |
| 2 — Phrase | ~120 chars |
| 3 — Idea | ~240 chars |
| **Combined** | **~460 chars** |

### Training Stages

| Stage | What trains | What's frozen | Active losses |
|-------|------------|--------------|---------------|
| 1 | Level 0 encoder | everything else | char |
| 2 | Level 1 + downsample | Level 0 | char, word-span |
| 3 | Level 2 + downsample | Levels 0-1 | char, word, contrastive |
| 4 | Level 3 + downsample | Levels 0-2 | char, word, contrastive |
| 5 | Decoder + output head | all encoder | char, word, contrastive |
| 6 | everything | nothing | all (low LR) |

## Usage

### Hierarchical (staged training)

```bash
# Full 6-stage training
python -m vm12.staged_trainer --text-path gutenberg_convos.txt --batch-size 32

# Start from a specific stage
python -m vm12.staged_trainer --text-path gutenberg_convos.txt --start-stage 3

# Custom steps per stage
python -m vm12.staged_trainer --text-path data.txt --stage-steps 50000,30000,20000,20000,30000,50000

# Fresh start (no checkpoint resume)
python -m vm12.staged_trainer --text-path data.txt --no-resume
```

### Legacy (flat VM12)

```bash
# Original single-stage training
python -m vm12.train --text-path gutenberg_convos.txt --batch-size 32

# Quick test
python -m vm12.train --text-path training_data.txt --max-steps 200 --no-resume
```

### Encoding

```bash
python -m vm12.encode "Hello, World!"
```

## Files

| File | Purpose |
|------|---------|
| `vm12/config.py` | Per-level configs, loss weights, contrastive params |
| `vm12/hierarchy.py` | `HierarchicalModel` — encoder, decoder, U-Net skips, dense gradients |
| `vm12/losses.py` | `WordSpanLoss`, `InfoNCELoss`, `HierarchicalLoss` |
| `vm12/staged_trainer.py` | 6-stage training pipeline with freeze/unfreeze, CLI |
| `vm12/scanning_block.py` | Multi-scale Scanner (Conv1d, kernel sizes 1-K) |
| `vm12/model.py` | Legacy `VM12Model` (flat baseline) |
| `vm12/encode.py` | 99-char encoding, `<START>`, `<INPUT>`, `</INPUT>` tokens |
| `vm12/data.py` | Sequence datasets (conversation + plain text) |
| `vm12/train.py` | Legacy single-stage training loop |
| `scripts/generate_data.py` | Gemini API synthetic conversation generator |

## Key Design Decisions

### Why hierarchical?
The original VM12 had ~40 chars of context. Stacking 4 levels with stride-2 downsampling gives ~460 chars — enough for paragraph-level reasoning. Each level's scanner sees ~30 positions, but those positions represent increasingly abstract features.

### Why U-Net decoder?
Skip connections from encoder to decoder carry fine-grained detail. Without them, the decoder must reconstruct character-level information from a compressed idea vector — too lossy. With them, the decoder can focus on high-level decisions while low-level detail flows through the skips.

### Why staged training?
Training compression and reasoning from scratch simultaneously fails — gradients from upper levels are meaningless before lower levels have learned to compress. Freeze-and-stack (like WaveNet, VQ-VAE-2) lets each level learn its job before the next level builds on it.

### Why contrastive loss at upper levels?
Phrase and idea levels don't have clear next-token prediction targets. InfoNCE contrastive loss (adjacent passages = positive, different documents = negative) gives them a semantically meaningful objective without needing labels.

### Why stride-2 instead of stride-4?
Characters are information-dense — one character can change meaning ("bat" vs "cat"). Stride-2 with kernel-4 gives 50% overlap, so every character contributes to at least two positions at the next level. Safer than stride-4 for char-level models.

## Requirements

```
torch>=2.0
numpy>=1.24
```
