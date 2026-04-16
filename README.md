# Toric Markov Model

A PyTorch-based trading model that combines toric topology with Markov chains for cryptocurrency market prediction.

## Overview

This project implements a novel approach to financial time series prediction using:
- Toric (toroidal) topology for cyclic pattern recognition
- Markov chains for state transition modeling
- Fractal cell architecture for multi-scale feature extraction
- Continuous feature embeddings with quantization

## Features

- **Trading Model V3**: Advanced trading signal prediction with basis features
- **Pattern Detection**: Automatic identification of market patterns
- **Backtesting**: Comprehensive backtesting framework
- **Confidence Analysis**: Model confidence evaluation tools

## Installation

```bash
cd toric_markov_model
pip install -r requirements.txt
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm
- tensorboard
- datasets
- tiktoken

## Usage

### Training

```bash
python scripts/train_trading_v3_basis.py \
    --data_path btc_data_with_basis.csv \
    --checkpoint_dir checkpoints_trading_v3_basis \
    --epochs 10 \
    --batch_size 32
```

### Backtesting

```bash
python scripts/backtest_trading_v3.py \
    --data_path btc_data_with_basis.csv \
    --checkpoint_path checkpoints_trading_v3_basis/best_model.pt
```

### Confidence Analysis

```bash
python scripts/analyze_confidence.py \
    --data_path btc_data_with_basis.csv \
    --checkpoint_path checkpoints_trading_v3_basis/best_model.pt
```

## Project Structure

```
toric_markov_model/
├── toric_markov_model/
│   ├── model/          # Model implementations
│   ├── data/           # Dataset and data processing
│   ├── train/          # Training utilities
│   └── config/         # Configuration files
├── scripts/            # Training and evaluation scripts
├── tests/              # Unit tests
└── checkpoints/        # Model checkpoints (not tracked)
```

## Model Architecture

The core model (`ToricTradingModelV3`) features:
- Continuous feature embedding with quantization
- Fractal cell layers for hierarchical pattern extraction
- Toric topology for cyclic market behavior modeling
- Multi-head pattern detection

## Testing

```bash
pytest tests/
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
