"""Tokenizers for experiments."""

from __future__ import annotations


class SimpleTokenizer:
    """Very small tokenizer for local experiments."""

    def __init__(self, vocab_size: int = 10_000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = vocab_size - 1

    def encode(self, text: str, max_len: int) -> list[int]:
        words = text.strip().split()
        ids = [self._hash_word(w) for w in words]
        ids = ids[: max_len - 1]
        ids.append(self.eos_token_id)
        if len(ids) < max_len:
            ids += [self.pad_token_id] * (max_len - len(ids))
        return ids

    def _hash_word(self, word: str) -> int:
        if not word:
            return self.pad_token_id
        return 1 + (hash(word) % (self.vocab_size - 2))
