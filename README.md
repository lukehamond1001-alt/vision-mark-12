# Vision Mark 12

Convolutional LLM that predicts **all output characters in a single forward pass**.

No attention, no learned embeddings, no autoregressive loop. One pass in, entire response out.

**~1.6M parameters**

## Architecture

```
Input: "What is the capital of France?"
  → one-hot encoding (99-dim per character)
  ↓
Multi-Scale Scanner (Conv1d, kernel sizes 1-10, 3 filters each)
  30 features per position (3 filters × 10 scales)
  ↓
Projection: Conv1d(30 → 512) + ReLU
  ↓
Dilated Causal Blocks (residual connections):
  Block 1: Conv1d(512, 512, k=3, dilation=1)  + RMSNorm + ReLU
  Block 2: Conv1d(512, 512, k=3, dilation=2)  + RMSNorm + ReLU
  Block 3: Conv1d(512, 512, k=3, dilation=4)  + RMSNorm + ReLU
  Block 4: Conv1d(512, 512, k=3, dilation=8)  + RMSNorm + ReLU
  ↓
Output: Conv1d(512 → 99) at EVERY position
  ↓
"The capital of France is Paris."  ← predicted all at once
```

## What Makes This Different

- **Parallel generation** — predicts all output characters in one forward pass, not one-at-a-time
- **No attention** — uses multi-scale convolutions + dilated causal convolutions
- **Character-level** — works on raw characters (99 vocab), not subword tokens
- **No embeddings** — raw one-hot vectors, no learned embedding table
- **No autoregressive loop** — inference is O(1) forward passes, not O(n)
- **Iterative refinement** — optional 1-2 extra passes to polish output coherence
- **Dilated receptive field** — scanner (10 chars) + dilated convs (16 chars) = 26+ character context

## How Inference Works

```
Traditional LLM (transformer):
  "France?" → "T" → "Th" → "The" → ... → "The capital is Paris."
  N forward passes for N characters (slow)

Vision Mark 12 (convolutional):
  "France?" → "The capital is Paris."
  1 forward pass for ALL characters (fast)
```

The model outputs a prediction at every position simultaneously. Position i predicts character i+1. No feeding output back in. Optional refinement passes can improve coherence.

## Usage

```bash
# View encoding
python -m vm12.encode "Hello, World!"

# Train on conversation data
python -m vm12.train --text-path gutenberg_convos.txt --batch-size 32 --lr 0.001 --no-resume

# Train on plain text
python -m vm12.train --text-path training_data.txt --no-resume

# Resume training
python -m vm12.train --text-path gutenberg_convos.txt

# Custom context length
python -m vm12.train --text-path training_data.txt --context-len 512

# Quick test (200 steps)
python -m vm12.train --text-path training_data.txt --max-steps 200 --batch-size 8 --no-resume
```

## Files

| File | Purpose |
|------|---------|
| `vm12/encode.py` | 99-char encoding, `<START>`, `<INPUT>`, `</INPUT>` tokens |
| `vm12/scanning_block.py` | Multi-scale Scanner (Conv1d, kernel sizes 1-K) |
| `vm12/model.py` | VM12Model — scanner + dilated causal blocks + all-position output |
| `vm12/config.py` | Hyperparameters (dilation pattern, context length, etc.) |
| `vm12/data.py` | Sequence datasets with shifted targets (conversation + plain text) |
| `vm12/train.py` | Training loop, eval, one-pass generation, iterative refinement |
| `scripts/generate_data.py` | Gemini API synthetic conversation generator |

## Key Design Decisions

### Why all-position prediction?
Convolutions compute features at every position in parallel. Traditional models throw away all but the last position. We use all of them — every position predicts its next character. This means training gets N gradient signals per sequence instead of 1.

### Why dilated convolutions?
Stacking dilations [1, 2, 4, 8] with kernel size 3 gives an exponentially growing receptive field:
- Block 1: sees 3 positions
- Block 2: sees 7 positions
- Block 3: sees 15 positions
- Block 4: sees 31 positions

Combined with the scanner (10 positions), each output sees ~26 characters of context.

### Why iterative refinement?
In one-pass generation, position 5 doesn't know what position 4 predicted. This can cause incoherence. By feeding the first-pass output back in for 1-2 more passes, each position can see what its neighbors produced and refine accordingly.

## Requirements

```
torch>=2.0
numpy>=1.24
```
