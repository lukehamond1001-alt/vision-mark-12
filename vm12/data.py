"""Data pipeline for Vision Mark 12.

Supports two training modes:
1. Conversation mode: <INPUT>...</INPUT> delimited multi-turn conversations.
   Only response characters are predicted (masked loss).
2. Plain text mode: next-character prediction on raw text.

Both modes return full sequences with shifted targets at every position.
Auto-detects format based on whether the data contains <INPUT> markers.
"""

import os
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from vm12.encode import (
    encode, encode_chars, CHAR_TO_INT, VOCAB_SIZE,
    START_TOKEN, INPUT_TOKEN, END_INPUT_TOKEN
)


def _clean_text(text: str) -> str:
    """Normalize text to only use characters in our vocab."""
    replacements = {
        '\n': ' ', '\r': ' ', '\t': ' ',
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2014': '-', '\u2013': '-',
        '\ufeff': '',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text


def _strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    start_markers = ['*** START OF THE PROJECT GUTENBERG', '*** START OF THIS PROJECT GUTENBERG']
    end_markers = ['*** END OF THE PROJECT GUTENBERG', '*** END OF THIS PROJECT GUTENBERG']

    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            nl = text.find('\n')
            if nl != -1:
                text = text[nl + 1:]
            break

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


# ---------------------------------------------------------------------------
# Conversation mode
# ---------------------------------------------------------------------------

def parse_conversation_file(path: str) -> list[list[tuple[str, str]]]:
    """Parse a file with <INPUT>...</INPUT> delimited conversations.

    Returns list of conversations, each a list of (user_msg, response) pairs.
    """
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    return _parse_conversations(text)


def _parse_conversations(text: str) -> list[list[tuple[str, str]]]:
    chunks = re.split(r'\n\s*\n', text.strip())
    all_convos = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        turns = _parse_turns(chunk)
        if turns:
            all_convos.append(turns)
    return all_convos


def _parse_turns(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r'<INPUT>(.*?)</INPUT>(.*?)(?=<INPUT>|$)', re.DOTALL)
    matches = pattern.findall(text)
    turns = []
    for user_msg, response in matches:
        user_msg = user_msg.strip()
        response = _clean_text(response).strip()
        if user_msg and response:
            turns.append((user_msg, response))
    return turns


def _encode_full_conversation(turns: list[tuple[str, str]]) -> tuple[list[int], list[bool]]:
    """Encode a conversation into a flat sequence + response mask.

    Returns:
        ids: Full token sequence [START, INPUT, ...user..., /INPUT, ...response..., INPUT, ...]
        is_response: Boolean mask, True at positions that are response characters.
    """
    ids = [START_TOKEN]
    is_response = [False]

    for user_msg, response in turns:
        user_ids = encode_chars(user_msg)
        response_ids = encode_chars(response)

        # <INPUT>user</INPUT>
        ids.append(INPUT_TOKEN)
        is_response.append(False)
        ids.extend(user_ids)
        is_response.extend([False] * len(user_ids))
        ids.append(END_INPUT_TOKEN)
        is_response.append(False)

        # response characters
        ids.extend(response_ids)
        is_response.extend([True] * len(response_ids))

    return ids, is_response


class ConversationSeqDataset(Dataset):
    """Fixed-length chunks from conversations with response masks.

    Each sample is a (input_seq, target_seq, loss_mask) triple:
    - input_seq[i] is the character at position i
    - target_seq[i] is the character at position i+1 (shifted)
    - loss_mask[i] is 1 if position i+1 is a response character
    """

    def __init__(self, encoded_convos: list[tuple[list[int], list[bool]]],
                 context_len: int = 256):
        self.context_len = context_len
        self.chunks = []

        for ids, mask in encoded_convos:
            # Slide a window over the conversation
            for start in range(0, len(ids) - context_len, context_len // 2):
                end = start + context_len + 1  # +1 for the target at last position
                if end > len(ids):
                    break
                chunk_ids = ids[start:end]
                chunk_mask = mask[start:end]
                self.chunks.append((chunk_ids, chunk_mask))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids, mask = self.chunks[idx]
        input_seq = torch.tensor(ids[:-1], dtype=torch.long)
        target_seq = torch.tensor(ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.float)  # shifted — mask for targets
        return input_seq, target_seq, loss_mask


# ---------------------------------------------------------------------------
# Plain text mode
# ---------------------------------------------------------------------------

def load_text_corpus(path: str = None, max_chars: int = 5_000_000) -> str:
    """Load raw text for training, cleaned for our vocab."""
    if path and os.path.exists(path):
        p = Path(path)
        if p.is_dir():
            texts = []
            for f in sorted(p.glob("*.txt")):
                raw = f.read_text(encoding="utf-8", errors="ignore")
                raw = _strip_gutenberg(raw)
                raw = _clean_text(raw)
                texts.append(raw)
            text = " ".join(texts)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
            text = _strip_gutenberg(text)
            text = _clean_text(text)
        return text[:max_chars]

    dict_paths = ["/usr/share/dict/words", "/usr/share/dict/american-english"]
    for dp in dict_paths:
        if os.path.exists(dp):
            with open(dp, "r") as f:
                words = [w.strip() for w in f if w.strip().isalpha() and len(w.strip()) <= 20]
            random.shuffle(words)
            chunks = []
            i = 0
            while i < len(words) and len(" ".join(chunks)) < max_chars:
                sentence_len = random.randint(3, 12)
                sentence = " ".join(words[i:i + sentence_len])
                chunks.append(sentence + ".")
                i += sentence_len
            return " ".join(chunks)[:max_chars]

    return "The quick brown fox jumps over the lazy dog. " * 1000


class PlainTextSeqDataset(Dataset):
    """Fixed-length chunks for plain-text next-character prediction.

    Every position predicts the next character. No masking needed.
    """

    def __init__(self, encoded: list[int], context_len: int = 256):
        self.context_len = context_len
        self.data = encoded
        # Number of non-overlapping chunks
        self.n_chunks = max((len(encoded) - 1) // context_len, 0)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.context_len
        end = start + self.context_len + 1
        if end > len(self.data):
            end = len(self.data)
            start = end - self.context_len - 1

        chunk = self.data[start:end]
        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)
        return input_seq, target_seq


# ---------------------------------------------------------------------------
# Mixed dataset
# ---------------------------------------------------------------------------

class MixedSeqDataset(Dataset):
    """Combines conversation and plaintext datasets.

    All samples are normalized to 3-tuples: (input, target, mask).
    Plaintext samples get an all-ones mask (predict every position).
    """

    def __init__(self, convo_dataset=None, plaintext_dataset=None):
        self.convo = convo_dataset
        self.plain = plaintext_dataset
        self.convo_len = len(convo_dataset) if convo_dataset else 0
        self.plain_len = len(plaintext_dataset) if plaintext_dataset else 0

    def __len__(self):
        return self.convo_len + self.plain_len

    def __getitem__(self, idx):
        if idx < self.convo_len:
            return self.convo[idx]  # already (input, target, mask)
        else:
            plain_idx = idx - self.convo_len
            input_seq, target_seq = self.plain[plain_idx]
            # All-ones mask: predict every position
            mask = torch.ones_like(target_seq, dtype=torch.float)
            return input_seq, target_seq, mask


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def collate_seq(batch):
    """Collate for sequence datasets.

    Handles both plain text (input, target) and conversation (input, target, mask).
    """
    has_mask = len(batch[0]) == 3

    if has_mask:
        inputs, targets, masks = zip(*batch)
        return (torch.stack(inputs), torch.stack(targets), torch.stack(masks))
    else:
        inputs, targets = zip(*batch)
        return (torch.stack(inputs), torch.stack(targets))


def _is_conversation_file(path: Path) -> bool:
    """Check if a single file contains <INPUT>...</INPUT> markers."""
    try:
        head = path.read_text(encoding='utf-8', errors='ignore')[:2000]
        return '</INPUT>' in head
    except Exception:
        return False


def create_dataloaders(config, text_path: str = None,
                       val_split: float = 0.1, num_workers: int = 0):
    """Load data, auto-detect format, return train/val loaders.

    When text_path is a directory, each .txt file is classified independently
    as conversation or plaintext. Both types are loaded and combined into a
    single MixedSeqDataset.
    """
    p = Path(text_path) if text_path else None

    if p and p.is_dir():
        return _create_mixed_loaders(config, p, val_split, num_workers)
    elif p and p.exists() and _is_conversation_file(p):
        return _create_conversation_loaders(config, text_path, val_split, num_workers)
    else:
        return _create_plaintext_loaders(config, text_path, val_split, num_workers)


def _create_mixed_loaders(config, dir_path: Path, val_split, num_workers):
    """Load a directory with a mix of conversation and plaintext files."""
    convo_files = []
    plain_files = []

    for f in sorted(dir_path.glob("*.txt")):
        if _is_conversation_file(f):
            convo_files.append(f)
        else:
            plain_files.append(f)

    print(f"Directory {dir_path}: {len(convo_files)} conversation + {len(plain_files)} plaintext files")

    # --- Conversation data ---
    convo_train_ds = None
    convo_val_ds = None
    if convo_files:
        all_convos = []
        for f in convo_files:
            all_convos.extend(parse_conversation_file(str(f)))
        random.shuffle(all_convos)
        encoded = [_encode_full_conversation(c) for c in all_convos]

        split_idx = int(len(encoded) * (1 - val_split))
        convo_train_ds = ConversationSeqDataset(encoded[:split_idx], config.context_len)
        convo_val_ds = ConversationSeqDataset(encoded[split_idx:], config.context_len)

        total_turns = sum(len(c) for c in all_convos)
        total_chars = sum(len(ids) for ids, _ in encoded)
        print(f"  Conversations: {len(all_convos):,} ({total_turns:,} turns, {total_chars:,} chars)")
        print(f"  Convo chunks: {len(convo_train_ds):,} train, {len(convo_val_ds):,} val")

    # --- Plaintext data ---
    plain_train_ds = None
    plain_val_ds = None
    if plain_files:
        texts = []
        for f in plain_files:
            raw = f.read_text(encoding='utf-8', errors='ignore')
            raw = _strip_gutenberg(raw)
            raw = _clean_text(raw)
            texts.append(raw)
            print(f"  Plaintext {f.name}: {len(raw):,} chars")

        combined = ' '.join(texts)
        encoded_plain = encode(combined)

        split_idx = int(len(encoded_plain) * (1 - val_split))
        plain_train_ds = PlainTextSeqDataset(encoded_plain[:split_idx], config.context_len)
        plain_val_ds = PlainTextSeqDataset(encoded_plain[split_idx:], config.context_len)

        print(f"  Plaintext total: {len(combined):,} chars, {len(encoded_plain):,} tokens")
        print(f"  Plain chunks: {len(plain_train_ds):,} train, {len(plain_val_ds):,} val")

    # --- Combine ---
    train_ds = MixedSeqDataset(convo_train_ds, plain_train_ds)
    val_ds = MixedSeqDataset(convo_val_ds, plain_val_ds)

    print(f"Combined: {len(train_ds):,} train chunks, {len(val_ds):,} val chunks "
          f"(context_len={config.context_len})")

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_seq,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_seq,
        pin_memory=False, drop_last=True,
    )
    return train_loader, val_loader


def _create_conversation_loaders(config, text_path, val_split, num_workers):
    p = Path(text_path)
    all_convos = []
    if p.is_dir():
        for f in sorted(p.glob("*.txt")):
            all_convos.extend(parse_conversation_file(str(f)))
    else:
        all_convos = parse_conversation_file(text_path)

    random.shuffle(all_convos)

    # Encode all conversations
    encoded = [_encode_full_conversation(c) for c in all_convos]

    split_idx = int(len(encoded) * (1 - val_split))
    train_encoded = encoded[:split_idx]
    val_encoded = encoded[split_idx:]

    train_ds = ConversationSeqDataset(train_encoded, config.context_len)
    val_ds = ConversationSeqDataset(val_encoded, config.context_len)

    total_turns = sum(len(c) for c in all_convos)
    total_chars = sum(len(ids) for ids, _ in encoded)
    print(f"Conversations: {len(all_convos):,} ({total_turns:,} turns, {total_chars:,} total chars)")
    print(f"Train chunks: {len(train_ds):,} | Val chunks: {len(val_ds):,} "
          f"(context_len={config.context_len})")

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_seq,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_seq,
        pin_memory=False, drop_last=True,
    )
    return train_loader, val_loader


def _create_plaintext_loaders(config, text_path, val_split, num_workers):
    text = load_text_corpus(text_path)
    encoded = encode(text)
    print(f"Corpus: {len(text):,} characters, {len(encoded):,} encoded tokens")

    split_idx = int(len(encoded) * (1 - val_split))
    train_enc = encoded[:split_idx]
    val_enc = encoded[split_idx:]

    train_ds = PlainTextSeqDataset(train_enc, config.context_len)
    val_ds = PlainTextSeqDataset(val_enc, config.context_len)
    print(f"Train chunks: {len(train_ds):,} | Val chunks: {len(val_ds):,} "
          f"(context_len={config.context_len})")

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_seq,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_seq,
        pin_memory=False, drop_last=True,
    )
    return train_loader, val_loader

