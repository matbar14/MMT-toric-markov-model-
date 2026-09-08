"""Versioned, strict checkpoint loading and atomic writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from ..data.trading_dataset_v3 import TradingDatasetV3
from ..model.trading_model_v3 import ToricTradingModelV3

FORMAT_VERSION = 2


def save_checkpoint(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint cannot be safely loaded; legacy checkpoints require retraining") from error
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format; retrain with the current code")
    model = ToricTradingModelV3(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if "decision_thresholds" in checkpoint:
        model.set_decision_thresholds(checkpoint["decision_thresholds"])
    if model.predict_return and not model.has_aux_stats.item():
        raise ValueError("checkpoint is missing training auxiliary statistics")
    model.to(device).eval()
    return checkpoint, model


def dataset_from_checkpoint(checkpoint, data_path, split="test", return_aux_targets=False):
    normalization = {name: value.cpu().numpy() for name, value in checkpoint["normalization"].items()}
    auxiliary = {name: checkpoint["model_state_dict"][name].cpu().numpy()
                 for name in ("aux_target_mean", "aux_target_std")}
    dataset = TradingDatasetV3(
        data_path, **checkpoint["data_config"], split=split,
        split_boundaries=checkpoint["split_boundaries"], normalization_stats=normalization,
        aux_target_stats=auxiliary, return_aux_targets=return_aux_targets,
    )
    if dataset.feature_names != checkpoint["feature_names"]:
        raise ValueError("checkpoint feature schema does not match the dataset")
    return dataset
