"""Standalone research playground for the toric model core.

This package is intentionally independent from the trading pipeline. It holds:

- ``toric_core``       — a cleaned-up, mathematically pure toric encoder;
- ``baselines``        — GRU / LSTM / Transformer / DLinear / naive baselines;
- ``train_utils``      — shared windows, standardization and training loops;
- ``synthetic_torus``  — home-turf validation on Markov-switching rotations;
- ``ett_experiment``   — pilot on real seasonal data (ETTh1, 96 -> 96 protocol).

Run from the repository root, e.g.::

    python -m research.synthetic_torus --models mmt,gru --epochs 6
    python -m research.ett_experiment --models dlinear,gru,mmt
"""
