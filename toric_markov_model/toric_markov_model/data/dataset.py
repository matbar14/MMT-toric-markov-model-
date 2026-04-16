"""Dataset loaders (including FineWeb hooks)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import SimpleTokenizer


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int) -> None:
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class TextDataset(Dataset):
    def __init__(self, text_path: str, tokenizer: SimpleTokenizer, seq_len: int) -> None:
        path = Path(text_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {text_path}")

        with path.open("r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

        if not self.lines:
            raise ValueError("Text dataset is empty")

        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.lines[idx], self.seq_len)
        return torch.tensor(ids, dtype=torch.long)
