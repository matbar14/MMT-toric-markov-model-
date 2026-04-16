"""Experiment-specific configuration presets."""

from __future__ import annotations

from .default import Config


def small_debug_config() -> Config:
    cfg = Config()
    cfg.model.vocab_size = 2_000
    cfg.model.dim_angles = 64
    cfg.model.max_len = 64
    cfg.model.num_states = 64
    cfg.model.num_levels = 4
    cfg.model.num_layers = 2

    cfg.train.batch_size = 16
    cfg.train.max_steps = 200
    cfg.train.log_interval = 20

    cfg.data.mode = "random"
    cfg.data.num_samples = 2_000
    return cfg


def baseline_config() -> Config:
    return Config()
