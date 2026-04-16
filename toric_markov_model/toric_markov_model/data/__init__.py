"""Data utilities."""

from __future__ import annotations

from torch.utils.data import DataLoader

from .dataset import RandomTokenDataset, TextDataset
from .tokenizer import SimpleTokenizer


def create_dataloader(
    mode: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_samples: int = 20_000,
    text_path: str = "",
    shuffle: bool = True,
) -> DataLoader:
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)

    if mode == "text":
        dataset = TextDataset(text_path=text_path, tokenizer=tokenizer, seq_len=seq_len)
    else:
        dataset = RandomTokenDataset(
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_samples=num_samples,
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


__all__ = [
    "SimpleTokenizer",
    "RandomTokenDataset",
    "TextDataset",
    "create_dataloader",
]
