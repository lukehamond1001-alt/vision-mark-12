"""Character encoder for Vision Mark 12.

Every key on a standard keyboard gets a whole number.
One neuron per character. Built for training from scratch.

    Lowercase:  a=1  b=2  ... z=26
    Uppercase:  A=27 B=28 ... Z=52
    Digits:     0=53 1=54 ... 9=62
    Space:      63
    Symbols:    !=64 @=65 #=66 ... (see CHAR_TO_INT)
    <START>:    96   — beginning of sequence
    <INPUT>:    97   — human started talking
    </INPUT>:   98   — human stopped talking
    Padding:    0

    "Hello" -> [34, 5, 12, 12, 15]   -- 5 neurons
    "Hi 5!" -> [34, 9, 63, 57, 64]   -- 5 neurons
"""

import torch


# Build the full keyboard mapping
_id = 1
CHAR_TO_INT = {}

# a-z lowercase
for c in 'abcdefghijklmnopqrstuvwxyz':
    CHAR_TO_INT[c] = _id
    _id += 1

# A-Z uppercase
for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    CHAR_TO_INT[c] = _id
    _id += 1

# 0-9
for c in '0123456789':
    CHAR_TO_INT[c] = _id
    _id += 1

# space
CHAR_TO_INT[' '] = _id
_id += 1

# symbols (every standard keyboard symbol)
for c in '!@#$%^&*()-_=+[]{}\\|;:\'",.<>/?`~':
    CHAR_TO_INT[c] = _id
    _id += 1

# Special tokens
CHAR_TO_INT['<START>'] = _id
_id += 1

CHAR_TO_INT['<INPUT>'] = _id
_id += 1

CHAR_TO_INT['</INPUT>'] = _id
_id += 1

VOCAB_SIZE = _id  # total values including padding at 0

START_TOKEN = CHAR_TO_INT['<START>']
INPUT_TOKEN = CHAR_TO_INT['<INPUT>']
END_INPUT_TOKEN = CHAR_TO_INT['</INPUT>']

INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}
INT_TO_CHAR[0] = ''  # padding


def encode(text: str) -> list[int]:
    """Encode a string to a list of integers. One number per character.

    A <START> token is prepended.
    """
    return [START_TOKEN] + [CHAR_TO_INT.get(c, 0) for c in text]


def encode_input(text: str) -> list[int]:
    """Encode a user input with <START><INPUT>..text..</INPUT> wrapper.

    Used at inference time: the model sees the full human turn
    and generates the response character by character.
    """
    chars = [CHAR_TO_INT.get(c, 0) for c in text]
    return [START_TOKEN, INPUT_TOKEN] + chars + [END_INPUT_TOKEN]


def encode_chars(text: str) -> list[int]:
    """Encode raw characters without any special tokens."""
    return [CHAR_TO_INT.get(c, 0) for c in text]


def decode(ids: list[int]) -> str:
    """Decode integer list back to string."""
    return "".join(INT_TO_CHAR.get(i, '?') for i in ids)


def encode_to_tensor(text: str, context_len: int = None) -> torch.Tensor:
    """Encode text to a tensor. Optionally pad/truncate to context_len."""
    ids = encode(text)
    if context_len is not None:
        ids = ids[:context_len]
        ids = ids + [0] * (context_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def show_neurons(text: str) -> None:
    """Print the neuron generated for every input character."""
    ids = encode(text)
    print(f'Input: "{text}"')
    print(f"Neurons: {len(ids)}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print()
    labels = ['<START>'] + list(text)
    for i, (label, val) in enumerate(zip(labels, ids)):
        display = repr(label) if label == ' ' else label
        print(f"  position {i}: {display:>8s} -> neuron value {val}")


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, World! 123"
    show_neurons(text)
    print()
    t = encode_to_tensor(text)
    print(f"Tensor: {t}")
    print(f"Shape:  {t.shape}")
    print(f"\n<START> token:   {START_TOKEN}")
    print(f"<INPUT> token:   {INPUT_TOKEN}")
    print(f"</INPUT> token:  {END_INPUT_TOKEN}")
