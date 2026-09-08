#!/usr/bin/env python3
"""Report signal quality and fixed train-only gate baselines without threshold search."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from toric_markov_model.train.calibration import decision_metrics
from toric_markov_model.train.checkpoint import dataset_from_checkpoint, load_checkpoint
from toric_markov_model.train.selection import partition_validation


def tabular_features(dataset):
    windows = np.lib.stride_tricks.sliding_window_view(
        dataset.features, dataset.seq_len, axis=0)[:len(dataset)].transpose(0, 2, 1)
    return np.concatenate((windows[:, -1], windows.mean(1), windows.std(1),
                           windows[:, -1] - windows[:, 0]), axis=1)


def ranking_metrics(labels, scores):
    return dict(prevalence=float(labels.mean()),
                average_precision=float(average_precision_score(labels, scores)) if labels.any() else None,
                roc_auc=float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                score_quantiles=np.quantile(scores, [0, 0.1, 0.5, 0.9, 1]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-test", action="store_true", help="Test is diagnostic once inspected")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint, model = load_checkpoint(args.checkpoint)
    with open(args.data, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != checkpoint["data_sha256"]:
        raise ValueError("fixed-split diagnostics require the original training dataset")
    train = dataset_from_checkpoint(checkpoint, args.data, split="train")
    validation = dataset_from_checkpoint(checkpoint, args.data, split="validation")
    selection, calibration, metadata = partition_validation(
        validation, checkpoint["validation_partition"]["calibration_fraction"])
    if metadata != checkpoint["validation_partition"]:
        raise ValueError("validation partition mismatch")
    train_labels = train.patterns[train.seq_len:train.seq_len + len(train), :-1]
    features = tabular_features(train)
    events = train_labels.any(axis=1)
    baselines = dict(
        logistic=make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
        histogram_gradient_boosting=HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=7, l2_regularization=10, early_stopping=False, random_state=42),
    )
    for baseline in baselines.values():
        baseline.fit(features, events)
    report = dict(epoch=checkpoint["epoch"], decision_thresholds=checkpoint["decision_thresholds"],
                  validation_partition=metadata, train_pattern_support=train_labels.sum(0).tolist(),
                  baseline_features="last, mean, std, last-minus-first; fixed train-only fitting",
                  test_is_diagnostic=args.include_test, splits={})
    datasets = dict(selection=selection, calibration=calibration)
    if args.include_test:
        datasets["test"] = dataset_from_checkpoint(checkpoint, args.data, split="test")
    for split, dataset in datasets.items():
        conditional, gates, labels = [], [], []
        with torch.inference_mode():
            for batch, targets in DataLoader(dataset, batch_size=64):
                outputs = model(batch)
                conditional.append(outputs["pattern_logits"][:, :-1].sigmoid().numpy())
                gates.append(outputs["non_hold_logit"].sigmoid().numpy().reshape(-1))
                labels.append(targets[:, :-1].numpy())
        conditional, gates, labels = np.concatenate(conditional), np.concatenate(gates), np.concatenate(labels)
        events = labels.any(axis=1)
        metrics = decision_metrics(conditional, gates, labels, checkpoint["decision_thresholds"])
        metrics.update(first_entry=dataset.timestamps.iloc[dataset.seq_len].isoformat(),
                       last_target=dataset.timestamps.iloc[-1].isoformat(),
                       pattern_support=labels.sum(0).tolist(), neural_gate=ranking_metrics(events, gates))
        features = tabular_features(dataset)
        metrics["baseline_gates"] = {
            name: ranking_metrics(events, baseline.predict_proba(features)[:, 1])
            for name, baseline in baselines.items()
        }
        report["splits"][split] = metrics
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
