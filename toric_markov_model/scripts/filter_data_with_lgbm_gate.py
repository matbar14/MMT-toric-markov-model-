"""Filter training data using LightGBM gate to create clean pattern dataset.

This script:
1. Loads trained LightGBM gate model
2. Filters train/val data to keep only samples where gate predicts non-hold
3. Saves filtered datasets for neural network training
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from toric_markov_model.data.trading_dataset_v3_enhanced import TradingDatasetV3Enhanced


def parse_args():
    parser = argparse.ArgumentParser(description="Filter data with LightGBM gate")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data")
    parser.add_argument("--lgbm-model", type=str, required=True, help="Path to LightGBM model (.txt)")
    parser.add_argument("--lgbm-results", type=str, required=True, help="Path to LightGBM results (.json)")
    parser.add_argument("--gate-threshold", type=float, default=0.20, help="Gate threshold (default: 0.20 from LightGBM best F1)")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--min-pattern-profit", type=float, default=0.003)
    parser.add_argument("--output-train", type=str, default="train_filtered.npz", help="Output train file")
    parser.add_argument("--output-val", type=str, default="val_filtered.npz", help="Output val file")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("FILTER DATA WITH LIGHTGBM GATE")
    print("=" * 80)
    
    # Load LightGBM model
    print(f"\nLoading LightGBM model from {args.lgbm_model}...")
    lgbm_model = lgb.Booster(model_file=args.lgbm_model)
    
    # Load results to get best threshold
    with open(args.lgbm_results, 'r') as f:
        lgbm_results = json.load(f)
    
    best_threshold = lgbm_results.get('best_threshold', args.gate_threshold)
    print(f"Using gate threshold: {best_threshold:.2f}")
    print(f"  (LightGBM best F1: {lgbm_results.get('best_f1', 0):.2%})")
    print(f"  (Precision: {lgbm_results.get('best_precision', 0):.2%}, Recall: {lgbm_results.get('best_recall', 0):.2%})")
    
    # Load datasets
    print("\nLoading enhanced datasets...")
    train_dataset = TradingDatasetV3Enhanced(
        csv_path=args.data,
        train=True,
        train_split=args.train_split,
        prediction_horizon=args.prediction_horizon,
        min_pattern_profit=args.min_pattern_profit,
        verbose=True,
    )
    
    val_dataset = TradingDatasetV3Enhanced(
        csv_path=args.data,
        train=False,
        train_split=args.train_split,
        prediction_horizon=args.prediction_horizon,
        min_pattern_profit=args.min_pattern_profit,
        normalization_stats=train_dataset.get_normalization_stats(),
        aux_target_stats=train_dataset.get_aux_target_stats(),
        verbose=True,
    )
    
    # Filter train data
    print("\nFiltering train data...")
    X_train = train_dataset.features
    y_train_patterns = train_dataset.patterns
    y_train_gate = (y_train_patterns[:, :-1].sum(axis=1) > 0).astype(int)
    
    train_gate_probs = lgbm_model.predict(X_train)
    train_gate_pred = (train_gate_probs >= best_threshold).astype(int)
    train_mask = train_gate_pred == 1
    
    X_train_filtered = X_train[train_mask]
    y_train_patterns_filtered = y_train_patterns[train_mask]
    
    print(f"  Original: {len(X_train)} samples")
    print(f"  Filtered: {len(X_train_filtered)} samples ({100 * len(X_train_filtered) / len(X_train):.1f}%)")
    print(f"  True non-hold in filtered: {y_train_gate[train_mask].sum()} ({100 * y_train_gate[train_mask].mean():.1f}%)")
    
    # Filter val data
    print("\nFiltering val data...")
    X_val = val_dataset.features
    y_val_patterns = val_dataset.patterns
    y_val_gate = (y_val_patterns[:, :-1].sum(axis=1) > 0).astype(int)
    
    val_gate_probs = lgbm_model.predict(X_val)
    val_gate_pred = (val_gate_probs >= best_threshold).astype(int)
    val_mask = val_gate_pred == 1
    
    X_val_filtered = X_val[val_mask]
    y_val_patterns_filtered = y_val_patterns[val_mask]
    
    print(f"  Original: {len(X_val)} samples")
    print(f"  Filtered: {len(X_val_filtered)} samples ({100 * len(X_val_filtered) / len(X_val):.1f}%)")
    print(f"  True non-hold in filtered: {y_val_gate[val_mask].sum()} ({100 * y_val_gate[val_mask].mean():.1f}%)")
    
    # Pattern distribution in filtered data
    print("\nPattern distribution in filtered train data:")
    pattern_names = [
        "Bullish Div", "Bearish Div", "CVD Rev Bull", "CVD Rev Bear",
        "CVD Exh Bull", "CVD Exh Bear", "CVD SF Bull", "CVD SF Bear",
        "CVD Spike Bull", "CVD Spike Bear", "Basis Long", "Basis Short",
        "Accumulation", "Distribution", "POC Break Up", "POC Break Down", "Hold",
    ]
    
    for i, name in enumerate(pattern_names):
        count = int(y_train_patterns_filtered[:, i].sum())
        pct = 100 * count / len(y_train_patterns_filtered)
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Save filtered data
    print(f"\nSaving filtered train data to {args.output_train}...")
    np.savez_compressed(
        args.output_train,
        features=X_train_filtered,
        patterns=y_train_patterns_filtered,
        normalization_stats=train_dataset.get_normalization_stats(),
        aux_target_stats=train_dataset.get_aux_target_stats(),
    )
    
    print(f"Saving filtered val data to {args.output_val}...")
    np.savez_compressed(
        args.output_val,
        features=X_val_filtered,
        patterns=y_val_patterns_filtered,
    )
    
    print("\n✓ Done! Filtered datasets saved.")
    print("\nNext steps:")
    print("1. Train neural network on filtered data WITHOUT gate head")
    print("2. Use LightGBM gate + neural pattern classifier for inference")
    print("=" * 80)


if __name__ == "__main__":
    main()
