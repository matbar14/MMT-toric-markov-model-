#!/usr/bin/env python3
"""Train the V3 contract using train/validation only; reserve test for backtesting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train import select_device, set_seed
from toric_markov_model.train.checkpoint import FORMAT_VERSION, load_checkpoint, save_checkpoint
from toric_markov_model.train.trading import build_pos_weight, configure_stage_trainability, run_epoch
from toric_markov_model.train.selection import improves_loss, partition_validation


def parse_args(native=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--volume-profile-period", type=int, default=20)
    parser.add_argument("--min-pattern-profit", type=float, default=0.003)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dim-angles", type=int, default=64)
    parser.add_argument("--num-states", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-pos-weight", type=float, default=5.0)
    parser.add_argument("--pos-weight-mode", choices=("ratio", "sqrt", "log"), default="sqrt")
    parser.add_argument("--aux-loss-weight", type=float, default=0.01)
    parser.add_argument("--pattern-threshold", type=float, default=0.5)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--resume-from", default="", help="Strict warm-start, not optimizer resume")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints_trading_v3_basis")
    if native:
        parser.add_argument("--binary", default=str(Path(__file__).resolve().parents[1] / "cpp/build/toric_train"))
        parser.add_argument("--threads", type=int, default=1)
        parser.add_argument("--prepare-only", action="store_true", help="Export train/validation tensors without running C++")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.batch_size, args.early_stop_patience) < 1 or args.num_workers < 0:
        raise ValueError("invalid epoch, batch, patience or worker count")
    if args.lr <= 0 or args.weight_decay < 0 or args.aux_loss_weight < 0:
        raise ValueError("invalid optimization parameters")
    if any(not 0 <= value <= 1 for value in
           (args.pattern_threshold, args.gate_threshold, args.confidence_threshold)):
        raise ValueError("decision thresholds must be in [0, 1]")
    if args.stage == 2 and not args.resume_from:
        raise ValueError("stage 2 requires a trained encoder/gate checkpoint via --resume-from")
    set_seed(args.seed)
    device = select_device(args.device)
    data_config = dict(seq_len=args.seq_len, prediction_horizon=args.prediction_horizon,
                       train_split=args.train_split, validation_split=args.validation_split,
                       min_pattern_profit=args.min_pattern_profit,
                       volume_profile_period=args.volume_profile_period)
    train_dataset = TradingDatasetV3(args.data, **data_config, split="train", return_aux_targets=True)
    normalization = train_dataset.get_normalization_stats()
    auxiliary = train_dataset.get_aux_target_stats()
    validation_dataset = TradingDatasetV3(
        args.data, **data_config, split="validation", return_aux_targets=True,
        split_boundaries=train_dataset.split_boundaries,
        normalization_stats=normalization, aux_target_stats=auxiliary,
    )
    validation_dataset, _, validation_partition = partition_validation(validation_dataset, args.calibration_fraction)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, drop_last=False)
    model = ToricTradingModelV3(
        num_features=train_dataset.features.shape[1], dim_angles=args.dim_angles,
        max_len=args.seq_len, num_states=args.num_states, num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    model.set_aux_target_stats(auxiliary)
    hasher = hashlib.sha256()
    with open(args.data, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if args.resume_from:
        previous, loaded = load_checkpoint(args.resume_from, device)
        if (previous["model_config"] != model.config or previous["data_config"] != data_config
                or previous["data_sha256"] != digest):
            raise ValueError("warm-start requires identical architecture, data and preprocessing")
        if previous.get("validation_partition") != validation_partition or "threshold_calibration" in previous:
            raise ValueError("warm-start requires the same untouched calibration partition")
        if args.stage == 2 and previous["stage"] not in (0, 1, 2):
            raise ValueError("stage 2 requires a trained event gate")
        model = loaded
    configure_stage_trainability(model, args.stage)
    labels = torch.from_numpy(train_dataset.patterns[args.seq_len:args.seq_len + len(train_dataset), :-1])
    events = labels.any(1)
    if not events.any() or events.all():
        raise ValueError("training requires both event and hold examples")
    pos_weight = build_pos_weight(labels[events], args.max_pos_weight, args.pos_weight_mode).to(device)
    gate_weight = build_pos_weight(events[:, None].float(), args.max_pos_weight, args.pos_weight_mode).to(device)
    print("Pattern support:", labels.sum(0).int().tolist())
    thresholds = dict(pattern_prob_threshold=args.pattern_threshold,
                      confidence_threshold=args.confidence_threshold, gate_threshold=args.gate_threshold)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    best_metric = float("inf")
    stale_epochs = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, pos_weight, gate_weight, args.stage,
                                   args.aux_loss_weight, optimizer=optimizer, thresholds=thresholds)
        val_metrics = run_epoch(model, validation_loader, device, pos_weight, gate_weight, args.stage,
                                 args.aux_loss_weight, thresholds=thresholds)
        metric_name = "loss"
        metric = val_metrics[metric_name]
        scheduler.step(metric)
        print(json.dumps({"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}), flush=True)
        payload = dict(
            format_version=FORMAT_VERSION, epoch=epoch + 1, stage=args.stage,
            model_config=model.config, model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            normalization={name: torch.as_tensor(value) for name, value in normalization.items()},
            data_config=data_config, split_boundaries=train_dataset.split_boundaries,
            data_sha256=digest, feature_names=train_dataset.feature_names,
            decision_thresholds=thresholds, validation_metrics=val_metrics, args=vars(args),
            validation_partition=validation_partition, selection_metric="loss", selection_mode="min",
        )
        save_checkpoint(payload, checkpoint_dir / f"last_model_stage{args.stage}.pt")
        if improves_loss(metric, best_metric):
            best_metric = metric
            stale_epochs = 0
            save_checkpoint(payload, checkpoint_dir / f"best_model_stage{args.stage}.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stop_patience:
            break
    print(f"Finished on {device}. Best validation {metric_name}={best_metric:.4f}. Test was not evaluated.")


if __name__ == "__main__":
    main()
