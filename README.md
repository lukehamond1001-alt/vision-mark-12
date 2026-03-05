# Vision Mark 12

Convolutional LLM that predicts **all output characters in a single forward pass**.

No attention, no learned embeddings, no autoregressive loop. One pass in, entire response out.

**3.23M parameters**

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
- **Dilated receptive field** — scanner (10 chars) + dilated convs (30 chars) = 40 character context

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

## Training Results

Trained on 1,576 conversations (3,200 turns, 865K response characters) from Project Gutenberg formatted as `<INPUT>...</INPUT>` conversations. Apple M4 (MPS), batch size 32, context length 256.

| Step | Train Loss | Train Acc | Val Loss | Val Acc |
|------|-----------|-----------|----------|---------|
| 100 | 2.449 | 33.9% | — | — |
| 500 | 0.773 | 78.5% | 1.850 | 54.9% |
| 1000 | 0.195 | 94.4% | 2.484 | 54.0% |
| 2000 | 0.097 | 97.5% | 3.124 | 55.1% |
| 4000 | 0.069 | 98.0% | 3.468 | 54.7% |
| 6000 | 0.052 | 98.4% | 3.808 | 54.4% |
| 8500 | 0.041 | 98.7% | 4.195 | 54.1% |

### Observations

- **Train accuracy rapidly reaches 98.7%** — the model memorizes the training data effectively
- **Val accuracy plateaus at ~54%** — significant overfitting, which is expected with 865K chars and 3.23M params
- **Train loss still decreasing** at step 8,500 — model continues to compress training data
- **One-pass generation** produces character sequences but lacks coherence — the core mechanism works but needs more data and architectural refinements

### Generation Samples (step 8,500)

```
1-pass:  "What is the capital of France?" → "A.tsl 000H0000600000RNRRYN..."
refined: "What is the capital of France?" → "Ardsyoemmovr huesautI  txe..."

1-pass:  "Tell me about dogs" → " tW4e0 000000000YN00R0AYY..."
refined: "Tell me about dogs" → " youesion,dtsooehtnmgbnwf..."
```

One-pass produces garbled output. Refinement passes produce more character-like text but without semantic meaning. The architecture successfully generates 40 characters in a single forward pass — the parallel prediction mechanism works — but coherent language generation requires larger receptive fields and more training data.

### Key Takeaways

1. **Parallel prediction works** — all positions fire simultaneously in one pass
2. **Overfitting is the bottleneck** — 3.23M params on 865K chars leads to memorization
3. **40-char receptive field is too small** for sentence-level coherence
4. **Next steps**: larger dilation patterns, more data, or deeper stacks to extend the receptive field

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
