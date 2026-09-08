"""Tensor-only interchange for the native trainer; preprocessing stays in Python."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ..model.trading_model_v3 import ToricTradingModelV3
from .checkpoint import FORMAT_VERSION, save_checkpoint
from .trading import build_pos_weight


def archive_name(name):
    return "weight__" + name.replace(".", "__")


class TensorArchive(torch.nn.Module):
    """TorchScript is a tensor container here, not the training graph."""

    def __init__(self, tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value.detach().cpu().contiguous().clone())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def export_bundle(model, train_dataset, validation_dataset, path, *, stage=0,
                  max_pos_weight=5.0, pos_weight_mode="sqrt", aux_loss_weight=0.01,
                  thresholds=None):
    if not model.predict_return:
        raise ValueError("native trainer currently requires all four auxiliary targets")
    if stage not in (0, 1, 2):
        raise ValueError("invalid training stage")
    if train_dataset.split != "train" or validation_dataset.split != "validation":
        raise ValueError("export accepts only train and validation, never test")
    if train_dataset.feature_names != validation_dataset.feature_names:
        raise ValueError("feature schema mismatch")
    model_config = model.config
    if model_config["max_len"] != train_dataset.seq_len:
        raise ValueError("model and dataset sequence lengths must match")
    dimensions = [model_config[name] for name in (
        "num_features", "dim_angles", "max_len", "num_states", "num_levels",
        "num_layers", "n_bits", "use_attention", "num_patterns",
    )]
    tensors = {archive_name(name): value for name, value in model.named_parameters()}
    thresholds = thresholds or dict(pattern_prob_threshold=0.5, confidence_threshold=0.0, gate_threshold=0.5)
    tensors.update(
        bundle_version=torch.tensor(1), stage=torch.tensor(stage),
        model_dimensions=torch.tensor(dimensions, dtype=torch.int64),
        dropout=torch.tensor(model_config["dropout"], dtype=torch.float64),
        loss_settings=torch.tensor([aux_loss_weight, thresholds["pattern_prob_threshold"],
                                    thresholds["confidence_threshold"], thresholds["gate_threshold"]],
                                   dtype=torch.float64),
    )
    for prefix, dataset in (("train", train_dataset), ("validation", validation_dataset)):
        target_slice = slice(dataset.seq_len, dataset.seq_len + len(dataset))
        tensors[prefix + "_features"] = torch.from_numpy(dataset.features)
        tensors[prefix + "_labels"] = torch.from_numpy(dataset.patterns[target_slice])
        tensors[prefix + "_auxiliary"] = torch.from_numpy(dataset.aux_targets[target_slice])
    labels = tensors["train_labels"][:, :-1]
    events = labels.any(1)
    if not events.any() or events.all():
        raise ValueError("training requires event and hold examples")
    tensors["positive_weight"] = build_pos_weight(labels[events], max_pos_weight, pos_weight_mode)
    tensors["gate_weight"] = build_pos_weight(events[:, None].float(), max_pos_weight, pos_weight_mode)
    archive = torch.jit.script(TensorArchive(tensors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.jit.save(archive, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return archive


def import_weights(model, path):
    archive = torch.jit.load(str(path), map_location="cpu")
    exported = dict(archive.named_buffers())
    expected = {archive_name(name) for name, _ in model.named_parameters()}
    actual = {name for name in exported if name.startswith("weight__")}
    if actual != expected:
        raise ValueError("native weights do not match the Python model schema")
    state = model.state_dict()
    for name, parameter in model.named_parameters():
        value = exported[archive_name(name)]
        if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
            raise ValueError(f"invalid native parameter {name}")
        state[name] = value
    model.load_state_dict(state, strict=True)
    return archive


def convert_checkpoint(template_path, native_path, metrics_path, destination):
    template = torch.load(template_path, map_location="cpu", weights_only=True)
    if template.get("format_version") != FORMAT_VERSION:
        raise ValueError("invalid checkpoint template version")
    model = ToricTradingModelV3(**template["model_config"])
    model.load_state_dict(template["model_state_dict"], strict=True)
    archive = import_weights(model, native_path)
    epoch = int(archive.epoch.item())
    with open(metrics_path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    record = next(item for item in records if item["epoch"] == epoch)
    metrics = record["validation"]
    for prefix, counts in (("pattern", archive.validation_pattern_counts.sum(1)),
                            ("gate", archive.validation_gate_counts)):
        positive, false_positive, false_negative = counts.tolist()
        metrics[prefix + "_precision"] = positive / max(1, positive + false_positive)
        metrics[prefix + "_recall"] = positive / max(1, positive + false_negative)
    metrics["per_pattern_counts"] = archive.validation_pattern_counts.tolist()
    stage = template["stage"]
    metrics["loss"] = (metrics["gate_loss"] if stage == 1 else metrics["pattern_loss"] if stage == 2
                       else metrics["pattern_loss"] + metrics["gate_loss"] +
                       template["args"]["aux_loss_weight"] * metrics["aux_loss"])
    template.update(model_state_dict=model.state_dict(), epoch=epoch, validation_metrics=metrics,
                    backend="libtorch-cpp", optimizer_resume_supported=False)
    save_checkpoint(template, Path(destination))
    return template
