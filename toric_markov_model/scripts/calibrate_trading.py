#!/usr/bin/env python3
"""Fit thresholds on the reserved validation tail; never read test windows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from toric_markov_model.train.calibration import fit_thresholds
from toric_markov_model.train.checkpoint import dataset_from_checkpoint, load_checkpoint, save_checkpoint
from toric_markov_model.train.selection import partition_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-signals", type=int, default=20)
    parser.add_argument("--max-signal-rate", type=float, default=0.5)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("use a new output path; do not overwrite a selected checkpoint")
    checkpoint, model = load_checkpoint(args.checkpoint)
    if "threshold_calibration" in checkpoint:
        raise ValueError("calibrate the selected training checkpoint, not an already calibrated copy")
    metadata = checkpoint.get("validation_partition")
    if metadata is None:
        raise ValueError("checkpoint has no reserved calibration partition; retrain first")
    with open(args.data, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != checkpoint["data_sha256"]:
        raise ValueError("calibration requires the original training dataset")
    validation = dataset_from_checkpoint(checkpoint, args.data, split="validation")
    _, calibration, actual = partition_validation(validation, metadata["calibration_fraction"])
    if actual != metadata:
        raise ValueError("calibration boundaries differ from the training protocol")
    conditional, gates, labels = [], [], []
    with torch.inference_mode():
        for features, targets in DataLoader(calibration, batch_size=args.batch_size):
            outputs = model(features)
            conditional.append(outputs["pattern_logits"][:, :-1].sigmoid().numpy())
            gates.append(outputs["non_hold_logit"].sigmoid().numpy().reshape(-1))
            labels.append(targets[:, :-1].numpy())
    report = fit_thresholds(np.concatenate(conditional), np.concatenate(gates),
                            np.concatenate(labels), min_signals=args.min_signals,
                            max_signal_rate=args.max_signal_rate)
    report.update(validation_partition=metadata, source_checkpoint=str(Path(args.checkpoint).resolve()),
                  data_sha256=digest, test_used=False, probability_calibrated=False,
                  production_approved=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if report["accepted"]:
        checkpoint["decision_thresholds"] = report["decision_thresholds"]
        checkpoint["threshold_calibration"] = {key: value for key, value in report.items() if key != "candidates"}
        save_checkpoint(checkpoint, output)
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))
    if not report["accepted"]:
        print("No eligible thresholds beat the constant-pattern baseline; no checkpoint written.")


if __name__ == "__main__":
    main()
