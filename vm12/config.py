"""Configuration for Vision Mark 12 — Hierarchical Convolutional LLM."""

from dataclasses import dataclass, field


@dataclass
class LevelConfig:
    """Configuration for a single encoder/decoder level."""

    width: int = 512                  # channel width for this level
    kernel_size: int = 3              # kernel size for dilated blocks
    dilation_pattern: list[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Scanner settings (only used at Level 0)
    pairs_per_fan_in: int = 3         # conv filters per kernel size
    max_fan_in: int = 10              # kernel sizes 1..K

    # Downsampling to next level
    downsample_kernel: int = 4
    downsample_stride: int = 2        # stride-2 with kernel-4 = 50% overlap


@dataclass
class VM12Config:
    """All hyperparameters for the Hierarchical Convolutional LLM."""

    vocab_size: int = 99              # 0=pad, 1-26=a-z, 27-52=A-Z, 53-62=digits, 63=space, 64-95=symbols, 96=START, 97=INPUT, 98=END_INPUT
    space_token: int = 63             # token id for space character

    # Number of hierarchy levels (0=char, 1=word, 2=phrase, 3=idea)
    num_levels: int = 4

    # Per-level configs — defaults give 4 identical levels
    level_configs: list[LevelConfig] = field(default_factory=lambda: [
        LevelConfig(width=512, dilation_pattern=[1, 2, 4, 8]),     # Level 0: char
        LevelConfig(width=512, dilation_pattern=[1, 2, 4, 8]),     # Level 1: word
        LevelConfig(width=512, dilation_pattern=[1, 2, 4, 8]),     # Level 2: phrase
        LevelConfig(width=512, dilation_pattern=[1, 2, 4, 8]),     # Level 3: idea
    ])

    # Sequence lengths
    context_len: int = 256            # fixed context window for training
    max_gen_len: int = 200            # max chars to generate

    # Loss weights (used in joint fine-tuning, Stage 6)
    char_loss_weight: float = 1.0
    word_loss_weight: float = 0.5
    phrase_loss_weight: float = 0.3
    idea_loss_weight: float = 0.2

    # Contrastive loss settings
    contrastive_dim: int = 128        # projection head output dim
    contrastive_temp: float = 0.07    # InfoNCE temperature
    min_negatives: int = 256          # minimum negative samples per batch

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_steps: int = 200_000
    grad_clip: float = 1.0

    # Staged training LR multipliers (relative to base lr)
    stage_lr_multipliers: list[float] = field(default_factory=lambda: [
        1.0,   # Stage 1: baseline VM12
        1.0,   # Stage 2: word level
        1.0,   # Stage 3: phrase level
        1.0,   # Stage 4: idea level
        0.5,   # Stage 5: decoder
        0.1,   # Stage 6: joint fine-tuning
    ])

    # Logging / checkpointing
    log_every: int = 100
    val_every: int = 500
    save_every: int = 2000
