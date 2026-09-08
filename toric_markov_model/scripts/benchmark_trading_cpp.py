#!/usr/bin/env python3
"""Compare warm CPU epochs on the same prepared tensors and initial weights."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train.cpp_bridge import import_weights
from toric_markov_model.train.trading import configure_stage_trainability, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="native/ directory from --prepare-only")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "cpp/build/toric_train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3, help="Measured epochs after one warmup epoch")
    args = parser.parse_args()
    if min(args.batch_size, args.threads, args.repeats) <= 0:
        raise ValueError("batch size, threads and repeats must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    bundle = torch.jit.load(str(args.bundle_dir / "input.pt"), map_location="cpu")
    template = torch.load(args.bundle_dir / "template.pt", map_location="cpu", weights_only=True)
    stage = int(bundle.stage.item())
    config = template["model_config"]
    thresholds = template["decision_thresholds"]
    training_args = template["args"]
    model = ToricTradingModelV3(**config)
    model.load_state_dict(template["model_state_dict"], strict=True)
    import_weights(model, args.bundle_dir / "input.pt")
    configure_stage_trainability(model, stage)
    loaders = []
    for name in ("train", "validation"):
        features = getattr(bundle, name + "_features")
        labels = getattr(bundle, name + "_labels")
        auxiliary = getattr(bundle, name + "_auxiliary")
        windows = features.unfold(0, config["max_len"], 1).permute(0, 2, 1)[:len(labels)]
        dataset = TensorDataset(windows, labels, auxiliary)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=(name == "train"), drop_last=False,
                                  generator=torch.Generator().manual_seed(training_args["seed"])))
    torch.manual_seed(training_args["seed"])
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=training_args["lr"], weight_decay=training_args["weight_decay"])
    python_times = []
    for epoch in range(args.repeats + 1):
        start = time.perf_counter()
        run_epoch(model, loaders[0], "cpu", bundle.positive_weight, bundle.gate_weight, stage,
                  training_args["aux_loss_weight"], optimizer=optimizer, thresholds=thresholds)
        run_epoch(model, loaders[1], "cpu", bundle.positive_weight, bundle.gate_weight, stage,
                  training_args["aux_loss_weight"], thresholds=thresholds)
        if epoch > 0:
            python_times.append(time.perf_counter() - start)
    with tempfile.TemporaryDirectory(prefix="toric-cpp-benchmark-") as directory:
        command = [str(args.binary.resolve()), "--input", str((args.bundle_dir / "input.pt").resolve()),
                   "--output-dir", directory, "--epochs", str(args.repeats + 1),
                   "--batch-size", str(args.batch_size), "--threads", str(args.threads),
                   "--patience", str(args.repeats + 2), "--stage", str(stage),
                   "--lr", str(training_args["lr"]), "--weight-decay", str(training_args["weight_decay"]),
                   "--seed", str(training_args["seed"])]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        with (Path(directory) / "metrics.jsonl").open() as stream:
            records = [json.loads(line) for line in stream]
        cpp_times = [record["train"]["seconds"] + record["validation"]["seconds"] for record in records[1:]]
    python_median = statistics.median(python_times)
    cpp_median = statistics.median(cpp_times)
    print(json.dumps(dict(
        device="cpu", threads=args.threads, batch_size=args.batch_size, repeats=args.repeats,
        train_samples=len(loaders[0].dataset), validation_samples=len(loaders[1].dataset), model_config=config,
        python_seconds=python_times, cpp_seconds=cpp_times,
        python_median_seconds=python_median, cpp_median_seconds=cpp_median,
        speedup=python_median / cpp_median,
        scope="train plus validation on prepared tensors; excludes CSV, startup, checkpoint I/O and warmup",
        caveat="independent RNG streams and optimizer implementations; not a quality comparison",
    ), indent=2))


if __name__ == "__main__":
    main()
