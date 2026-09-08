#!/usr/bin/env python3
"""Inspect the experimental Toric forecast; never submits orders."""

import argparse
import json

import pandas as pd
import torch

from toric_markov_model.train.payoff_checkpoint import forecast_closed_bars, load_payoff_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV containing completed candles only")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint, model, _ = load_payoff_checkpoint(args.checkpoint)
    print(json.dumps(forecast_closed_bars(checkpoint, model, pd.read_csv(args.data)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
