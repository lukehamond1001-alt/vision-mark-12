"""Training loop for Vision Mark 12.

All-position next-character prediction with dilated causal convolutions.
One forward pass predicts the next char at every position simultaneously.
Supports MPS, CUDA, and CPU. AdamW optimizer, gradient clipping.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from vm12.config import VM12Config
from vm12.model import VM12Model
from vm12.data import create_dataloaders
from vm12.encode import decode, encode_input, encode_chars, START_TOKEN, INT_TO_CHAR


CHECKPOINT_DIR = Path("checkpoints")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: VM12Model, val_loader, device: torch.device,
             max_batches: int = 50) -> tuple[float, float]:
    """Returns (val_loss, val_accuracy) across all positions."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    num_batches = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        has_mask = len(batch) == 3
        if has_mask:
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            mask = None

        logits = model(x)  # (batch, seq_len, vocab)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction='none')
        loss = loss.reshape_as(y)

        if mask is not None:
            n = mask.sum()
            if n > 0:
                loss = (loss * mask).sum() / n
            else:
                loss = loss.mean()
        else:
            loss = loss.mean()

        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=-1)
        if mask is not None:
            total_correct += ((preds == y) * mask).sum().item()
            total_count += mask.sum().item()
        else:
            total_correct += (preds == y).sum().item()
            total_count += y.numel()

    model.train()
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_count, 1)
    return avg_loss, accuracy


@torch.no_grad()
def demo_prediction(model: VM12Model, text: str, device: torch.device,
                    n_chars: int = 40) -> str:
    """Generate n_chars in ONE forward pass.

    Feeds the prompt, then reads predicted characters from positions
    after the prompt — no autoregressive loop.
    """
    model.eval()
    ids = encode_input(text)
    prompt_len = len(ids)

    # Pad input with zeros for the generation positions
    full_len = prompt_len + n_chars
    full_ids = ids + [0] * n_chars
    x = torch.tensor([full_ids], dtype=torch.long, device=device)

    logits = model(x)  # (1, full_len, vocab) — ONE forward pass

    # Read predictions at positions prompt_len-1 through prompt_len+n_chars-2
    # Position i predicts char at i+1, so position prompt_len-1 predicts first gen char
    generated = []
    for i in range(n_chars):
        pred_pos = prompt_len - 1 + i
        if pred_pos >= logits.size(1):
            break
        pred_idx = logits[0, pred_pos].argmax().item()
        generated.append(pred_idx)

    model.train()
    return decode(generated)


@torch.no_grad()
def demo_prediction_refined(model: VM12Model, text: str, device: torch.device,
                            n_chars: int = 40, n_refine: int = 2) -> str:
    """Generate with iterative refinement.

    Pass 0: one-shot generation (same as demo_prediction)
    Pass 1..n_refine: feed the generated text back in to refine predictions.
    """
    model.eval()
    ids = encode_input(text)
    prompt_len = len(ids)

    # Pass 0: initial generation
    full_ids = ids + [0] * n_chars
    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(x)

    generated = []
    for i in range(n_chars):
        pred_pos = prompt_len - 1 + i
        if pred_pos >= logits.size(1):
            break
        generated.append(logits[0, pred_pos].argmax().item())

    # Refinement passes
    for _ in range(n_refine):
        refined_ids = ids + generated
        x = torch.tensor([refined_ids], dtype=torch.long, device=device)
        logits = model(x)

        new_generated = []
        for i in range(len(generated)):
            pred_pos = prompt_len - 1 + i
            if pred_pos >= logits.size(1):
                break
            new_generated.append(logits[0, pred_pos].argmax().item())
        generated = new_generated

    model.train()
    return decode(generated)


def train(config: VM12Config, text_path: str = None, resume: bool = True):
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = create_dataloaders(config, text_path=text_path)

    model = VM12Model(config).to(device)
    print(model.layer_info())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    start_step = 0
    best_val_loss = float("inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = CHECKPOINT_DIR / "latest.pt"

    if resume and latest_path.exists():
        print(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at step {start_step}")

    model.train()
    step = start_step
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    t0 = time.time()

    print(f"\nStarting training from step {step}...")
    print(f"LR={config.lr}, batch_size={config.batch_size}, "
          f"context_len={config.context_len}, grad_clip={config.grad_clip}\n")

    demo_prompts = [
        "What is the capital of France?",
        "Tell me about dogs",
        "How does the sun work?",
        "What is 2+2?",
    ]

    while step < config.max_steps:
        for batch in train_loader:
            if step >= config.max_steps:
                break

            has_mask = len(batch) == 3
            if has_mask:
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                mask = None

            logits = model(x)  # (batch, seq_len, vocab)

            # All-position cross entropy
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   reduction='none')
            loss = loss.reshape_as(y)

            if mask is not None:
                n = mask.sum()
                if n > 0:
                    loss_scalar = (loss * mask).sum() / n
                else:
                    loss_scalar = loss.mean()
            else:
                loss_scalar = loss.mean()

            loss_scalar.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # Tracking
            running_loss += loss_scalar.item()
            preds = logits.argmax(dim=-1)
            if mask is not None:
                running_correct += ((preds == y) * mask).sum().item()
                running_count += mask.sum().item()
            else:
                running_correct += (preds == y).sum().item()
                running_count += y.numel()
            step += 1

            if step % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                accuracy = running_correct / max(running_count, 1)
                elapsed = time.time() - t0
                print(
                    f"Step {step:>6d} | loss: {avg_loss:.4f} | "
                    f"acc: {accuracy:.3f} | "
                    f"elapsed: {elapsed:.0f}s"
                )
                running_loss = 0.0
                running_correct = 0
                running_count = 0

            if step % config.val_every == 0:
                val_loss, val_acc = evaluate(model, val_loader, device)
                print(f"  >>> val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(model, optimizer, step, best_val_loss, config,
                                     CHECKPOINT_DIR / "best.pt")
                    print(f"  >>> New best! Saved best.pt")

                for prompt in demo_prompts:
                    result = demo_prediction(model, prompt, device)
                    refined = demo_prediction_refined(model, prompt, device)
                    print(f'  >>> 1-pass: "{prompt}" → "{result}"')
                    print(f'  >>>  refine:  → "{refined}"')

            if step % config.save_every == 0:
                _save_checkpoint(model, optimizer, step, best_val_loss, config,
                                 latest_path)
                print(f"  >>> Saved latest.pt at step {step}")

    _save_checkpoint(model, optimizer, step, best_val_loss, config,
                     CHECKPOINT_DIR / "final.pt")
    print(f"\nTraining complete at step {step}. Saved final.pt")


def _save_checkpoint(model, optimizer, step, best_val_loss, config, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Vision Mark 12")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--text-path", type=str, default=None,
                        help="Path to text file or directory of .txt files")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = VM12Config(
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        context_len=args.context_len,
    )

    train(config, text_path=args.text_path, resume=not args.no_resume)


if __name__ == "__main__":
    main()
