#!/usr/bin/env python3
"""Analyze confidence and score distribution for V3 pattern detector."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="btc_data_with_basis.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_trading_v3_basis/best_model.pt")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--pattern-prob-threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]

    normalization = checkpoint.get("normalization")
    norm_stats = None
    if normalization is not None:
        norm_stats = {
            "feature_mean": normalization["feature_mean"].detach().cpu().numpy(),
            "feature_std": normalization["feature_std"].detach().cpu().numpy(),
        }

    test_dataset = TradingDatasetV3(
        csv_path=args.data,
        seq_len=ckpt_args.seq_len,
        prediction_horizon=ckpt_args.prediction_horizon,
        train=False,
        min_pattern_profit=getattr(ckpt_args, "min_pattern_profit", 0.005),
        normalization_stats=norm_stats,
    )

    model = ToricTradingModelV3(
        num_features=test_dataset.features.shape[1],
        dim_angles=ckpt_args.dim_angles,
        max_len=ckpt_args.seq_len,
        num_states=ckpt_args.num_states,
        num_levels=4,
        num_layers=ckpt_args.num_layers,
        n_bits=8,
        use_attention=True,
        num_patterns=17,
        predict_return=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Analyzing confidence distribution...")
    print("=" * 80)

    pattern_names = [
        "Bullish Div", "Bearish Div", "CVD Rev Bull", "CVD Rev Bear",
        "CVD Exh Bull", "CVD Exh Bear", "CVD SF Bull", "CVD SF Bear",
        "CVD Spike Bull", "CVD Spike Bear", "Basis Long", "Basis Short",
        "Accumulation", "Distribution", "POC Break Up", "POC Break Down", "Hold",
    ]
    bullish_patterns = {0, 2, 4, 6, 8, 10, 12, 14}

    confidences = []
    scores = []
    patterns_detected = []
    hold_scores = []

    sample_limit = min(args.samples, len(test_dataset))
    for i in range(sample_limit):
        features, _ = test_dataset[i]
        features = features.unsqueeze(0).to(device)

        with torch.no_grad():
            result = model.detect_patterns(
                features,
                confidence_threshold=args.confidence_threshold,
                pattern_prob_threshold=args.pattern_prob_threshold,
            )

        pattern_idx = int(result["strongest_pattern"].item())
        confidence = float(result["strongest_confidence"].item())
        best_score = float(result["best_non_hold_score"].item())
        hold_score = float(result["hold_score"].item())

        patterns_detected.append(pattern_idx)
        confidences.append(confidence)
        scores.append(best_score)
        hold_scores.append(hold_score)

        if i < 10:
            print(
                f"Sample {i}: Pattern={pattern_names[pattern_idx]}, "
                f"Conf={confidence:.3f}, Score={best_score:.3f}, HoldScore={hold_score:.3f}"
            )

    confidences = np.array(confidences)
    scores = np.array(scores)
    hold_scores = np.array(hold_scores)
    patterns_detected = np.array(patterns_detected)

    print(f"\nConfidence Statistics ({sample_limit} samples):")
    print(f"  Mean: {confidences.mean():.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Min: {confidences.min():.3f}")
    print(f"  Max: {confidences.max():.3f}")

    print(f"\nSignal Score Statistics ({sample_limit} samples):")
    print(f"  Mean non-hold score: {scores.mean():.3f}")
    print(f"  Mean hold score: {hold_scores.mean():.3f}")
    print(f"  Mean score margin (non-hold - hold): {(scores - hold_scores).mean():.3f}")

    bullish_mask = np.array([idx in bullish_patterns for idx in patterns_detected])
    print(f"\nBullish patterns with conf > 0.3: {np.sum(bullish_mask & (confidences > 0.3))}")
    print(f"Bullish patterns with conf > 0.5: {np.sum(bullish_mask & (confidences > 0.5))}")
    print(f"Hold pattern count: {np.sum(patterns_detected == 16)}")


if __name__ == "__main__":
    main()
