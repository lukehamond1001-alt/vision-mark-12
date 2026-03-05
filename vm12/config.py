"""Configuration for Vision Mark 12 — Convolutional LLM."""

from dataclasses import dataclass, field


@dataclass
class VM12Config:
    """All hyperparameters for the Vision Mark 12 architecture."""

    vocab_size: int = 99              # 0=pad, 1-26=a-z, 27-52=A-Z, 53-62=digits, 63=space, 64-95=symbols, 96=START, 97=INPUT, 98=END_INPUT

    # Scanner settings
    pairs_per_fan_in: int = 3         # conv filters per kernel size
    word_max_fan_in: int = 10         # kernel sizes 1..10

    # Dilated causal dense layers
    dense_width: int = 512
    dense_kernel_size: int = 3        # kernel size for dilated layers
    dilation_pattern: list[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Sequence lengths
    context_len: int = 256            # fixed context window for training
    max_gen_len: int = 200            # max chars to generate in one pass

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_steps: int = 200_000
    grad_clip: float = 1.0

    # Logging / checkpointing
    log_every: int = 100
    val_every: int = 500
    save_every: int = 2000
