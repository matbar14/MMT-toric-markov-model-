"""Shared windowing, standardization and training utilities for the experiments."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn


def make_windows(array: np.ndarray, seq_len: int, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding windows from (length, channels): returns x (N, T, C), y (N, H, C)."""
    if array.ndim != 2:
        raise ValueError("expected a (length, channels) array")
    total = seq_len + horizon
    if array.shape[0] < total:
        raise ValueError("series shorter than one window")
    frames = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).unfold(
        0, total, 1
    )  # (N, C, total)
    x = frames[..., :seq_len].transpose(1, 2).contiguous()
    y = frames[..., seq_len:].transpose(1, 2).contiguous()
    return x, y


class Standardizer:
    """Per-channel z-score statistics fitted on the training slice only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        if np.any(std <= 0) or not np.isfinite(std).all():
            raise ValueError("invalid standard deviation")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, array: np.ndarray) -> "Standardizer":
        return cls(array.mean(axis=0), array.std(axis=0) + 1e-8)

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / self.std


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 3,
    seed: int = 0,
    log: bool = True,
) -> dict:
    """Plain AdamW + MSE training with early stopping on validation MSE.

    Returns the fitted model (best validation state restored) and a history dict.
    """
    generator = torch.Generator().manual_seed(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_train = train_x.shape[0]
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = train_x[idx], train_y[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()

        val_mse, val_mae = evaluate(model, val_x, val_y)
        history.append(
            {"epoch": epoch, "train_mse": total_loss / batches,
             "val_mse": val_mse, "val_mae": val_mae}
        )
        if log:
            print(
                f"    epoch {epoch:02d}  train {total_loss / batches:.4f}"
                f"  val mse {val_mse:.4f}  mae {val_mae:.4f}",
                flush=True,
            )
        if val_mse < best_val - 1e-5:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            if log:
                print(f"    early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history.append({"train_seconds": round(time.time() - started, 1)})
    return {"model": model, "history": history, "best_val_mse": best_val}


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> tuple[float, float]:
    """MSE / MAE on the standardized scale, batched to bound memory."""
    model.eval()
    preds = []
    for start in range(0, x.shape[0], batch_size):
        preds.append(model(x[start : start + batch_size]))
    pred = torch.cat(preds, dim=0)
    mse = nn.functional.mse_loss(pred, y).item()
    mae = (pred - y).abs().mean().item()
    return mse, mae


@torch.no_grad()
def evaluate_naive(predict_fn, x: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> tuple[float, float]:
    """Metrics for a stateless predictor mapping an x-batch to a y-batch."""
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    count = 0
    for start in range(0, x.shape[0], batch_size):
        pred = predict_fn(x[start : start + batch_size])
        diff = pred - y[start : start + batch_size]
        sq_err_sum += diff.pow(2).sum().item()
        abs_err_sum += diff.abs().sum().item()
        count += diff.numel()
    return sq_err_sum / count, abs_err_sum / count


class _CorrectionView(nn.Module):
    """Exposes only the toric correction path of a HybridForecaster."""

    def __init__(self, hybrid) -> None:
        super().__init__()
        self.hybrid = hybrid

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.hybrid.encoder(features)
        return self.hybrid.correction_head(hidden).view(
            features.shape[0], self.hybrid.horizon, self.hybrid.num_channels
        )


def train_hybrid_residual(
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    base_epochs: int = 10,
    residual_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 4,
    seed: int = 0,
    log: bool = True,
) -> dict:
    """Two-phase hybrid training (roadmap item 2).

    Phase 1 trains the linear base (trend + seasonal heads) alone; the toric
    path is frozen and zero-initialized, so this is exactly a DLinear.
    Phase 2 freezes the base and trains the toric encoder + correction head
    against the base's *residual* (y - base(x)), so the toric core is asked a
    clean question — "explain what linearity cannot" — and can never
    destabilize the base with correction noise.

    Returns the same dict shape as ``train_model``; ``best_val_mse`` is the
    full-model validation MSE after phase 2.
    """
    history: list[dict] = []

    # ---- Phase 1: linear base only ----
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.correction_head.parameters():
        parameter.requires_grad_(False)
    if log:
        print("  [phase 1: linear base]", flush=True)
    base_outcome = train_model(
        model, train_x, train_y, val_x, val_y,
        epochs=base_epochs, batch_size=batch_size, lr=lr,
        patience=patience, seed=seed, log=log,
    )
    history += [{"phase": "base", **entry} for entry in base_outcome["history"]]

    # ---- Phase 2: toric correction on the frozen base's residual ----
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.correction_head.parameters():
        parameter.requires_grad_(True)
    for parameter in list(model.trend_head.parameters()) + list(model.seasonal_head.parameters()):
        parameter.requires_grad_(False)

    with torch.no_grad():
        model.eval()
        residual_train = train_y - model.linear_base(train_x)
        residual_val = val_y - model.linear_base(val_x)
    if log:
        print(
            f"  [phase 2: toric correction, residual val mse="
            f"{residual_val.pow(2).mean().item():.4f}]",
            flush=True,
        )
    view = _CorrectionView(model)
    residual_outcome = train_model(
        view, train_x, residual_train, val_x, residual_val,
        epochs=residual_epochs, batch_size=batch_size, lr=lr,
        patience=patience, seed=seed, log=log,
    )
    history += [{"phase": "residual", **entry} for entry in residual_outcome["history"]]

    model.eval()
    full_val_mse, full_val_mae = evaluate(model, val_x, val_y)
    return {"model": model, "history": history, "best_val_mse": full_val_mse}
