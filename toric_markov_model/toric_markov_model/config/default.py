"""Default project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    dim_angles: int = 128
    max_len: int = 256
    num_states: int = 256
    num_levels: int = 4
    num_layers: int = 3
    n_bits: int = 8
    use_attention: bool = True
    use_continuous: bool = True
    chunk_size: int = 32  # 0 = no chunking, >0 = chunk size for parallel processing


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 2_000
    grad_clip: float = 1.0
    log_interval: int = 100
    seed: int = 42
    device: str = "cuda"


@dataclass
class DataConfig:
    mode: str = "random"  # random | text
    text_path: str = ""
    num_samples: int = 20_000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
