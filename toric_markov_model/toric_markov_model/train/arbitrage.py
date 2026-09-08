"""Chronological training and strict research checkpoint schema for paired returns."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..arbitrage import ArbitrageConfig, pair_outcome
from ..model.arbitrage import ToricArbitrageModel
from .checkpoint import save_checkpoint
from .payoff import tabular_context


def targets(data, entries):
    return np.array([pair_outcome(data, int(entry))["net_return"] for entry in entries], dtype=np.float32)[:, None]


def fit_pair_models(data, train_entries, selection_entries, seq_len=32, epochs=12, seed=42):
    if epochs < 1 or len(train_entries) < 20 or not len(selection_entries):
        raise ValueError("positive epochs, at least 20 train entries and nonempty selection required")
    train_windows = data.windows(train_entries, seq_len)
    selection_windows = torch.from_numpy(data.windows(selection_entries, seq_len))
    if train_entries[-1] + data.config.horizon > selection_entries[0] - seq_len:
        raise ValueError("train targets must precede selection context")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train_targets = torch.from_numpy(targets(data, train_entries))
    selection_targets = torch.from_numpy(targets(data, selection_entries))
    model = ToricArbitrageModel(data.features.shape[1], max_len=seq_len)
    model.fit_statistics(data.features[train_entries[0] - seq_len:train_entries[-1]], train_targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_windows), train_targets), batch_size=32,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for features, target in loader:
            optimizer.zero_grad()
            loss = (model(features) - (target - model.target_mean) / model.target_std).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite paired training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = torch.cat([model.predict_pair_return(batch) for batch in selection_windows.split(32)])
            value = (prediction - selection_targets).square().mean().item()
        if not np.isfinite(value):
            raise ValueError("nonfinite selection loss")
        history.append(dict(epoch=epoch, selection_mse=value))
        if value < best_loss - 1e-10:
            best_loss, best_state, stale = value, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100))
    ridge.fit(tabular_context(train_windows), train_targets.numpy().ravel())
    return model.eval(), ridge, dict(history=history, selected_mse=best_loss,
                                    train_mean=float(train_targets.mean()), seed=seed)


def forecasts(model, ridge, windows):
    if model.training:
        raise ValueError("evaluation-mode model required")
    with torch.inference_mode():
        prediction = torch.cat([model.predict_pair_return(batch) for batch in torch.from_numpy(windows).split(32)])
    result = dict(toric=prediction.numpy().ravel(), ridge=ridge.predict(tabular_context(windows)))
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("nonfinite paired forecasts")
    return result


def save_pair_checkpoint(model, data, path, policy, provenance):
    metadata = json.loads(json.dumps(dict(policy=policy, provenance=provenance), allow_nan=False))
    save_checkpoint(dict(format="toric_arbitrage_v1", model_config=model.config,
                         model_state_dict=model.state_dict(), feature_names=data.feature_names,
                         execution=asdict(data.config), **metadata,
                         live_orders_allowed=False), path)


def load_pair_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "toric_arbitrage_v1" or checkpoint.get("live_orders_allowed") is not False:
        raise ValueError("only research arbitrage checkpoints can be loaded")
    config = ArbitrageConfig(**checkpoint["execution"])
    model = ToricArbitrageModel(**checkpoint["model_config"])
    state = checkpoint["model_state_dict"]
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite checkpoint tensors")
    model.load_state_dict(state, strict=True)
    if (not model.fitted_statistics.item() or (model.feature_std <= 0).any() or (model.target_std <= 0).any() or
            not isinstance(checkpoint["feature_names"], list) or
            len(checkpoint["feature_names"]) != model.num_features or
            not all(isinstance(name, str) and name for name in checkpoint["feature_names"]) or
            len(set(checkpoint["feature_names"])) != model.num_features or
            type(checkpoint["policy"].get("enabled")) is not bool):
        raise ValueError("invalid paired model statistics or feature schema")
    return checkpoint, model.eval(), config
