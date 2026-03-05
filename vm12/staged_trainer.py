"""Staged training for hierarchical convolutional LLM.

6 training stages, each building on the frozen weights from below:
  Stage 1: Train Level 0 (VM12 baseline) — char prediction only
  Stage 2: Freeze L0, train Level 1 — word-span loss
  Stage 3: Freeze L0-1, train Level 2 — contrastive loss
  Stage 4: Freeze L0-2, train Level 3 — contrastive loss
  Stage 5: Freeze encoder, train decoder — char + word + contrastive
  Stage 6: Unfreeze all, joint fine-tuning — low LR

Each stage manages its own optimizer, freeze/unfreeze logic,
active losses, and learning rate.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from vm12.config import VM12Config
from vm12.hierarchy import HierarchicalModel
from vm12.losses import HierarchicalLoss
from vm12.data import create_dataloaders
from vm12.encode import decode, encode_input


CHECKPOINT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_CONFIGS = {
    1: {
        'name': 'Baseline VM12 (Level 0)',
        'active_losses': {'char'},
        'freeze': [],           # nothing frozen
        'unfreeze': ['encoder_levels.0'],
        'lr_stage_idx': 0,
    },
    2: {
        'name': 'Word Scanner (Level 1)',
        'active_losses': {'char', 'word'},
        'freeze': ['encoder_levels.0'],
        'unfreeze': ['encoder_levels.1', 'downsample_blocks.0', 'word_head'],
        'lr_stage_idx': 1,
    },
    3: {
        'name': 'Phrase Scanner (Level 2)',
        'active_losses': {'char', 'word', 'contrastive'},
        'freeze': ['encoder_levels.0', 'encoder_levels.1', 'downsample_blocks.0'],
        'unfreeze': ['encoder_levels.2', 'downsample_blocks.1', 'contrastive_heads.2'],
        'lr_stage_idx': 2,
    },
    4: {
        'name': 'Idea Scanner (Level 3)',
        'active_losses': {'char', 'word', 'contrastive'},
        'freeze': ['encoder_levels.0', 'encoder_levels.1', 'encoder_levels.2',
                    'downsample_blocks.0', 'downsample_blocks.1'],
        'unfreeze': ['encoder_levels.3', 'downsample_blocks.2', 'contrastive_heads.3'],
        'lr_stage_idx': 3,
    },
    5: {
        'name': 'Decoder',
        'active_losses': {'char', 'word', 'contrastive'},
        'freeze': ['encoder_levels', 'downsample_blocks'],
        'unfreeze': ['decoder_levels', 'output_head', 'dense_skips'],
        'lr_stage_idx': 4,
    },
    6: {
        'name': 'Joint Fine-tuning',
        'active_losses': {'char', 'word', 'contrastive'},
        'freeze': [],
        'unfreeze': None,  # None = unfreeze everything
        'lr_stage_idx': 5,
    },
}


def _freeze_module(model: HierarchicalModel, prefix: str):
    """Freeze all parameters matching the given module prefix."""
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = False


def _unfreeze_module(model: HierarchicalModel, prefix: str):
    """Unfreeze all parameters matching the given module prefix."""
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = True


def apply_stage(model: HierarchicalModel, stage: int):
    """Apply freeze/unfreeze pattern for the given training stage."""
    cfg = STAGE_CONFIGS[stage]

    if cfg['unfreeze'] is None:
        # Stage 6: unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        return

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then selectively unfreeze
    for prefix in cfg['unfreeze']:
        _unfreeze_module(model, prefix)

    # Also unfreeze any modules from previous stages that should stay trainable
    # (for the output head which is needed at every stage after 5)
    if stage >= 5:
        _unfreeze_module(model, 'output_head')


def get_active_params(model: HierarchicalModel) -> list:
    """Return list of parameters with requires_grad=True."""
    return [p for p in model.parameters() if p.requires_grad]


def count_frozen(model: HierarchicalModel) -> tuple[int, int]:
    """Return (frozen_count, total_count) of parameters."""
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return frozen, total


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def demo_generation(model: HierarchicalModel, text: str, device: torch.device,
                    n_chars: int = 40) -> str:
    """Generate n_chars using the hierarchical model.

    Uses one-pass parallel prediction from the decoder output.
    """
    model.eval()
    ids = encode_input(text)
    prompt_len = len(ids)

    full_ids = ids + [0] * n_chars
    x = torch.tensor([full_ids], dtype=torch.long, device=device)

    output = model(x)
    logits = output['logits']  # (1, full_len, vocab)

    generated = []
    for i in range(n_chars):
        pred_pos = prompt_len - 1 + i
        if pred_pos >= logits.size(1):
            break
        pred_idx = logits[0, pred_pos].argmax().item()
        generated.append(pred_idx)

    model.train()
    return decode(generated)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: HierarchicalModel, val_loader, device: torch.device,
             active_losses: set, loss_fn: HierarchicalLoss,
             max_batches: int = 50) -> dict[str, float]:
    """Evaluate the model on the validation set."""
    model.eval()
    totals = {}
    num_batches = 0
    total_correct = 0
    total_count = 0

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

        output = model(x)
        losses = loss_fn(output, y, mask=mask, active_stages=active_losses)

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        num_batches += 1

        preds = output['logits'].argmax(dim=-1)
        if mask is not None:
            total_correct += ((preds == y) * mask).sum().item()
            total_count += mask.sum().item()
        else:
            total_correct += (preds == y).sum().item()
            total_count += y.numel()

    model.train()
    avg = {k: v / max(num_batches, 1) for k, v in totals.items()}
    avg['accuracy'] = total_correct / max(total_count, 1)
    return avg


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, stage, best_val_loss, config, path):
    """Save a training checkpoint."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "stage": stage,
        "best_val_loss": best_val_loss,
        "config": config,
    }, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Load a checkpoint. Returns (step, stage, best_val_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    if optimizer and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, KeyError):
            print("  Warning: could not restore optimizer state (param groups changed)")
    return ckpt.get('step', 0), ckpt.get('stage', 1), ckpt.get('best_val_loss', float('inf'))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_stage(model: HierarchicalModel, config: VM12Config,
                stage: int, train_loader, val_loader,
                device: torch.device, max_steps: int,
                start_step: int = 0, best_val_loss: float = float('inf')):
    """Train one stage of the hierarchy."""

    stage_cfg = STAGE_CONFIGS[stage]
    lr_mult = config.stage_lr_multipliers[stage_cfg['lr_stage_idx']]
    lr = config.lr * lr_mult

    print(f"\n{'='*60}")
    print(f"STAGE {stage}: {stage_cfg['name']}")
    print(f"{'='*60}")

    # Apply freeze/unfreeze
    apply_stage(model, stage)
    frozen, total = count_frozen(model)
    active_params = get_active_params(model)
    print(f"Parameters: {total:,} total, {frozen:,} frozen, {total-frozen:,} trainable")
    print(f"Active losses: {stage_cfg['active_losses']}")
    print(f"LR: {lr:.2e} (base={config.lr:.2e}, mult={lr_mult})")

    if not active_params:
        print("No trainable parameters! Skipping stage.")
        return start_step, best_val_loss

    # Fresh optimizer for each stage (only on trainable params)
    optimizer = torch.optim.AdamW(
        active_params, lr=lr, weight_decay=config.weight_decay
    )

    # Cosine LR schedule with warmup
    warmup_steps = min(500, max_steps // 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.01)

    loss_fn = HierarchicalLoss(config)

    model.train()
    step = start_step
    running_losses = {}
    running_correct = 0
    running_count = 0
    t0 = time.time()

    demo_prompts = [
        "What is the capital of France?",
        "Tell me about dogs",
    ]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stage_dir = CHECKPOINT_DIR / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting stage {stage} from step {step}, max {max_steps} steps...\n")

    end_step = start_step + max_steps
    while step < end_step:
        for batch in train_loader:
            if step >= end_step:
                break

            has_mask = len(batch) == 3
            if has_mask:
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                mask = None

            output = model(x)
            losses = loss_fn(output, y, mask=mask,
                             active_stages=stage_cfg['active_losses'])

            loss_total = losses['total']
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(active_params, config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # Warmup: linear ramp for first warmup_steps
            if step - start_step < warmup_steps:
                warmup_factor = (step - start_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr * warmup_factor
            else:
                scheduler.step()

            # Track losses
            for k, v in losses.items():
                running_losses[k] = running_losses.get(k, 0.0) + v.item()

            preds = output['logits'].argmax(dim=-1)
            if mask is not None:
                running_correct += ((preds == y) * mask).sum().item()
                running_count += mask.sum().item()
            else:
                running_correct += (preds == y).sum().item()
                running_count += y.numel()
            step += 1

            # Logging
            if step % config.log_every == 0:
                acc = running_correct / max(running_count, 1)
                elapsed = time.time() - t0
                loss_strs = [f"{k}={v/config.log_every:.4f}"
                             for k, v in sorted(running_losses.items())]
                cur_lr = optimizer.param_groups[0]['lr']
                print(
                    f"[S{stage}] Step {step:>6d} | "
                    f"{' | '.join(loss_strs)} | "
                    f"acc: {acc:.3f} | lr: {cur_lr:.2e} | "
                    f"elapsed: {elapsed:.0f}s"
                )
                running_losses = {}
                running_correct = 0
                running_count = 0

            # Validation
            if step % config.val_every == 0:
                val_metrics = evaluate(model, val_loader, device,
                                       stage_cfg['active_losses'], loss_fn)
                val_strs = [f"{k}={v:.4f}" for k, v in sorted(val_metrics.items())]
                print(f"  >>> VAL: {' | '.join(val_strs)}")

                val_loss = val_metrics.get('total', float('inf'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, stage,
                                     best_val_loss, config,
                                     stage_dir / "best.pt")
                    print(f"  >>> New best! Saved stage{stage}/best.pt")

                for prompt in demo_prompts:
                    result = demo_generation(model, prompt, device)
                    print(f'  >>> "{prompt}" → "{result}"')

            # Save checkpoint
            if step % config.save_every == 0:
                save_checkpoint(model, optimizer, step, stage,
                                 best_val_loss, config,
                                 stage_dir / "latest.pt")

    # Final save
    save_checkpoint(model, optimizer, step, stage,
                     best_val_loss, config,
                     stage_dir / "final.pt")
    print(f"\nStage {stage} complete at step {step}.")

    return step, best_val_loss


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_hierarchical(config: VM12Config, text_path: str = None,
                       start_stage: int = 1, resume: bool = True,
                       steps_per_stage: dict[int, int] = None):
    """Run the full staged training pipeline.

    Args:
        config: model/training config
        text_path: path to training data
        start_stage: stage to start from (1-6)
        resume: whether to resume from checkpoint
        steps_per_stage: dict mapping stage number to max steps
    """
    device = get_device()
    print(f"Device: {device}")

    if steps_per_stage is None:
        steps_per_stage = {
            1: 50_000,
            2: 30_000,
            3: 20_000,
            4: 20_000,
            5: 30_000,
            6: 50_000,
        }

    train_loader, val_loader = create_dataloaders(config, text_path=text_path)

    model = HierarchicalModel(config).to(device)
    print(model.layer_info())

    step = 0
    best_val_loss = float('inf')

    # Try to resume from a stage checkpoint
    if resume:
        for s in range(start_stage, 0, -1):
            stage_ckpt = CHECKPOINT_DIR / f"stage{s}" / "latest.pt"
            if stage_ckpt.exists():
                step, loaded_stage, best_val_loss = load_checkpoint(
                    stage_ckpt, model, device=device
                )
                print(f"Resumed from stage {loaded_stage}, step {step}")
                start_stage = loaded_stage
                break

    # Run each stage
    for stage in range(start_stage, 7):
        stage_steps = steps_per_stage.get(stage, 20_000)
        step, best_val_loss = train_stage(
            model, config, stage,
            train_loader, val_loader, device,
            max_steps=stage_steps,
            start_step=step,
            best_val_loss=best_val_loss,
        )

    # Final model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, None, step, 6, best_val_loss, config,
                     CHECKPOINT_DIR / "hierarchical_final.pt")
    print(f"\n{'='*60}")
    print(f"ALL STAGES COMPLETE. Final model saved.")
    print(f"Total steps: {step}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Hierarchical Convolutional LLM (staged)"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--text-path", type=str, default=None,
                        help="Path to text file or directory")
    parser.add_argument("--start-stage", type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6],
                        help="Stage to start training from")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stage-steps", type=str, default=None,
                        help="Comma-separated steps per stage, e.g. '50000,30000,20000,20000,30000,50000'")
    args = parser.parse_args()

    config = VM12Config(
        lr=args.lr,
        batch_size=args.batch_size,
        context_len=args.context_len,
    )

    steps_per_stage = None
    if args.stage_steps:
        vals = [int(s.strip()) for s in args.stage_steps.split(',')]
        steps_per_stage = {i + 1: v for i, v in enumerate(vals)}

    train_hierarchical(
        config,
        text_path=args.text_path,
        start_stage=args.start_stage,
        resume=not args.no_resume,
        steps_per_stage=steps_per_stage,
    )


if __name__ == "__main__":
    main()
