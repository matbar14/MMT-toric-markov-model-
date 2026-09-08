#!/usr/bin/env python3
"""Prepare the existing dataset, run native LibTorch training and convert checkpoints."""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path

import torch

from train_trading_v3_basis import parse_args
from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train import set_seed
from toric_markov_model.train.checkpoint import FORMAT_VERSION, load_checkpoint, save_checkpoint
from toric_markov_model.train.cpp_bridge import convert_checkpoint, export_bundle
from toric_markov_model.train.selection import partition_validation


def main():
    args = parse_args(native=True)
    if min(args.epochs, args.batch_size, args.early_stop_patience, args.threads) < 1:
        raise ValueError("epochs, batch, patience and threads must be positive")
    if args.num_workers != 0:
        raise ValueError("native trainer uses tensor batches, not Python DataLoader workers; omit --num-workers")
    for name in ("lr", "weight_decay", "aux_loss_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"invalid {name}")
    if args.stage == 2 and not args.resume_from:
        raise ValueError("stage 2 requires a trained checkpoint via --resume-from")
    binary = Path(args.binary).resolve()
    if not args.prepare_only and not binary.is_file():
        raise FileNotFoundError(f"build the native trainer with CMake first: {binary}")
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    output = Path(args.checkpoint_dir).resolve()
    native = output / "native"
    native.mkdir(parents=True, exist_ok=True)
    if any(native.iterdir()):
        raise ValueError("native output directory is not empty; use a new --checkpoint-dir for each run")
    data_config = dict(seq_len=args.seq_len, prediction_horizon=args.prediction_horizon,
                       train_split=args.train_split, validation_split=args.validation_split,
                       min_pattern_profit=args.min_pattern_profit,
                       volume_profile_period=args.volume_profile_period)
    train_dataset = TradingDatasetV3(args.data, **data_config, split="train", return_aux_targets=True)
    normalization = train_dataset.get_normalization_stats()
    auxiliary = train_dataset.get_aux_target_stats()
    validation = TradingDatasetV3(
        args.data, **data_config, split="validation", return_aux_targets=True,
        split_boundaries=train_dataset.split_boundaries,
        normalization_stats=normalization, aux_target_stats=auxiliary,
    )
    validation, _, validation_partition = partition_validation(validation, args.calibration_fraction)
    model = ToricTradingModelV3(
        num_features=train_dataset.features.shape[1], dim_angles=args.dim_angles,
        max_len=args.seq_len, num_states=args.num_states, num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.set_aux_target_stats(auxiliary)
    hasher = hashlib.sha256()
    with open(args.data, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if args.resume_from:
        previous, loaded = load_checkpoint(args.resume_from)
        if (previous["model_config"] != model.config or previous["data_config"] != data_config or
                previous["data_sha256"] != digest):
            raise ValueError("warm-start requires identical architecture, data and preprocessing")
        if previous.get("validation_partition") != validation_partition or "threshold_calibration" in previous:
            raise ValueError("warm-start requires the same untouched calibration partition")
        model = loaded
    thresholds = dict(pattern_prob_threshold=args.pattern_threshold,
                      confidence_threshold=args.confidence_threshold, gate_threshold=args.gate_threshold)
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("thresholds must be in [0, 1]")
    bundle_path = native / "input.pt"
    export_bundle(model, train_dataset, validation, bundle_path, stage=args.stage,
                  max_pos_weight=args.max_pos_weight, pos_weight_mode=args.pos_weight_mode,
                  aux_loss_weight=args.aux_loss_weight, thresholds=thresholds)
    template = dict(
        format_version=FORMAT_VERSION, stage=args.stage, model_config=model.config,
        model_state_dict=model.state_dict(),
        normalization={name: torch.as_tensor(value) for name, value in normalization.items()},
        data_config=data_config, split_boundaries=train_dataset.split_boundaries,
        data_sha256=digest, feature_names=train_dataset.feature_names,
        decision_thresholds=thresholds, args=vars(args),
        validation_partition=validation_partition, selection_metric="loss", selection_mode="min",
    )
    template_path = native / "template.pt"
    save_checkpoint(template, template_path)
    if args.prepare_only:
        print(f"Prepared {bundle_path}; only train/validation exported, no test data.")
        return
    command = [str(binary), "--input", str(bundle_path), "--output-dir", str(native),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--threads", str(args.threads), "--device", args.device, "--stage", str(args.stage),
               "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
               "--patience", str(args.early_stop_patience), "--seed", str(args.seed)]
    print("Starting native training (CSV preparation excluded from epoch timings).", flush=True)
    subprocess.run(command, check=True)
    for kind in ("best", "last"):
        destination = output / f"{kind}_model_stage{args.stage}.pt"
        convert_checkpoint(template_path, native / f"{kind}_weights.pt", native / "metrics.jsonl", destination)
        print(f"Saved Python-compatible checkpoint: {destination}")
    print("Native training finished. Test was not evaluated; optimizer warm-start only.")


if __name__ == "__main__":
    main()
