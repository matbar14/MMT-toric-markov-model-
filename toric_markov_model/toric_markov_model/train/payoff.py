"""Train and validate payoff forecasts without assuming a trading edge exists."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..execution import exit_on_bar
from ..model.payoff import ToricPayoffModel


def tabular_context(windows):
    return np.concatenate((windows[:, -1], windows.mean(1), windows.std(1),
                           windows[:, -1] - windows[:, 0]), axis=1)


def train_toric(data, partitions, seq_len=16, epochs=12, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train_entries, selection_entries = partitions["train"], partitions["selection"]
    model = ToricPayoffModel(len(data.feature_names), max_len=seq_len, dim_angles=16, num_states=8)
    target = torch.tensor(data.net_returns[train_entries], dtype=torch.float32)
    model.fit_statistics(data.features[train_entries[0] - seq_len:train_entries[-1]], target)
    loader = DataLoader(TensorDataset(torch.from_numpy(data.windows(train_entries, seq_len)), target),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    selection_features = torch.from_numpy(data.windows(selection_entries, seq_len))
    selection_targets = torch.tensor(data.net_returns[selection_entries], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    best, best_state, best_epoch, stale = float("inf"), None, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for features, targets in loader:
            optimizer.zero_grad()
            loss = ((model(features) - (targets - model.target_mean) / model.target_std) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite payoff training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(features)
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([model.predict_payoffs(batch) for batch in selection_features.split(64)])
            selection_loss = ((predictions - selection_targets) ** 2).mean().item()
        if not np.isfinite(selection_loss):
            raise ValueError("nonfinite payoff selection loss")
        scheduler.step(selection_loss)
        history.append(dict(epoch=epoch, train_normalized_mse=total / len(train_entries),
                            selection_mse=selection_loss))
        print(history[-1], flush=True)
        if selection_loss < best - 1e-9:
            best, best_state, best_epoch, stale = selection_loss, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_epoch=best_epoch, selection_mse=best, history=history)


def fit_baselines(windows, targets):
    features = tabular_context(windows)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100))
    ridge.fit(features, targets)
    trees = []
    for column in range(2):
        model = HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=50,
            l2_regularization=10, early_stopping=False, random_state=42,
        )
        model.fit(features, targets[:, column])
        trees.append(model)
    return dict(ridge=ridge, trees=trees, constant=targets.mean(0))


def predict_all(toric, baselines, windows):
    features = tabular_context(windows)
    with torch.inference_mode():
        neural = torch.cat([toric.predict_payoffs(batch)
                            for batch in torch.from_numpy(windows).split(64)]).numpy()
    return dict(toric=neural, ridge=baselines["ridge"].predict(features),
                trees=np.column_stack([model.predict(features) for model in baselines["trees"]]),
                constant=np.broadcast_to(baselines["constant"], (len(windows), 2)).copy())


def choose_actions(scores, threshold, allowed=None):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("finite [long, short] payoff scores required")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("nonnegative finite minimum net edge required")
    if allowed is not None:
        allowed = np.asarray(allowed)
        if allowed.shape != scores.shape or allowed.dtype != bool:
            raise ValueError("allowed sides must be an aligned boolean array")
        scores = np.where(allowed, scores, -np.inf)
    side = np.argmax(scores, axis=1)
    best = scores[np.arange(len(scores)), side]
    unequal = scores[:, 0] != scores[:, 1]
    return np.where((best > threshold) & unequal, np.where(side == 0, 1, -1), 0)


def simulate(data, entries, actions, config=None, *, stop_distances=None, take_distances=None,
             risk_fraction=None):
    """One position at a time, next-open fills, marked equity and adverse slippage."""
    config = config or data.config
    entries, actions = np.asarray(entries), np.asarray(actions)
    if (entries.ndim != 1 or not np.issubdtype(entries.dtype, np.integer) or
            actions.shape != entries.shape or not len(entries) or
            (np.diff(entries) <= 0).any() or not np.isin(actions, [-1, 0, 1]).all() or
            entries[0] < 0 or entries[-1] + config.horizon > len(data.frame)):
        raise ValueError("ordered complete entries and aligned actions required")
    distances = []
    for provided, default in ((stop_distances, config.stop_loss), (take_distances, config.take_profit)):
        values = np.full(len(entries), default) if provided is None else np.asarray(provided, dtype=float)
        if values.shape != entries.shape or not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
            raise ValueError("finite entry-aligned stop/take distances in (0, 1) required")
        distances.append(values)
    if risk_fraction is not None and (not np.isfinite(risk_fraction) or not 0 < risk_fraction < 1):
        raise ValueError("risk fraction must be finite and in (0, 1)")
    prices = data.frame[["spot_open", "spot_high", "spot_low", "spot_close"]].to_numpy()
    capital, side, next_allowed = 1.0, 0, int(entries[0])
    trade = None
    trades, equity = [], []
    position_by_entry = dict(zip(entries.tolist(), actions.tolist()))
    distances_by_entry = dict(zip(entries.tolist(), zip(*distances)))
    for bar in range(int(entries[0]), int(entries[-1] + config.horizon)):
        bar_open, bar_high, bar_low, bar_close = prices[bar]
        if side == 0 and bar >= next_allowed:
            side = position_by_entry.get(bar, 0)
            if side:
                entry_fill = bar_open * (1 + side * config.slippage)
                stop_distance, take_distance = distances_by_entry[bar]
                size_fraction = config.position_fraction
                if risk_fraction is not None:
                    stop_ratio = (1 - side * stop_distance) * (1 - side * config.slippage)
                    planned_loss = -side * (stop_ratio - 1) + config.fee * (1 + stop_ratio)
                    size_fraction = min(size_fraction, risk_fraction / planned_loss)
                notional = capital * size_fraction
                units = notional / entry_fill
                entry_fee = notional * config.fee
                capital -= entry_fee
                trade = dict(entry=bar, side=side, entry_fill=entry_fill, units=units,
                             entry_notional=notional, entry_fee=entry_fee, size_fraction=size_fraction,
                             stop_distance=float(stop_distance), take_distance=float(take_distance))
        if side:
            exited, raw_exit, reason = exit_on_bar(
                side, trade["entry_fill"], bar_open, bar_high, bar_low, bar_close,
                trade["stop_distance"], trade["take_distance"], bar - trade["entry"] + 1, config.horizon,
            )
            if exited:
                exit_fill = raw_exit * (1 - side * config.slippage)
                gross_pnl = side * trade["units"] * (exit_fill - trade["entry_fill"])
                exit_fee = trade["units"] * exit_fill * config.fee
                net_pnl = gross_pnl - trade["entry_fee"] - exit_fee
                capital += gross_pnl - exit_fee
                trades.append(dict(**trade, exit=bar, exit_fill=exit_fill, reason=reason,
                                   net_pnl=net_pnl, net_return=net_pnl / trade["entry_notional"],
                                   fees=trade["entry_fee"] + exit_fee))
                next_allowed, side = bar + config.cooldown + 1, 0
        marked = capital if side == 0 else capital + side * trade["units"] * (bar_close - trade["entry_fill"])
        equity.append(marked)
        if marked <= 0:
            raise ValueError("bankruptcy under configured sizing; no valid unlevered backtest")
    if side:
        raise ValueError("unfinished trade: evaluation must include the full exit horizon")
    curve = np.array(equity)
    returns = np.diff(np.r_[1.0, curve]) / np.r_[1.0, curve[:-1]]
    times = data.frame.timestamp.iloc[entries[0]:entries[-1] + config.horizon]
    daily_equity = pd.Series(curve, index=pd.DatetimeIndex(times)).resample("1D").last()
    daily_returns = daily_equity.pct_change().fillna(daily_equity.iloc[0] - 1).to_numpy()
    wins = sum(trade["net_pnl"] > 0 for trade in trades)
    profit = sum(max(trade["net_pnl"], 0) for trade in trades)
    loss = -sum(min(trade["net_pnl"], 0) for trade in trades)
    metrics = dict(samples=len(entries), signals=int((actions != 0).sum()), trades=len(trades),
                   total_return_pct=(capital - 1) * 100, win_rate=wins / max(len(trades), 1),
                   profit_factor=profit / loss if loss else None,
                   max_drawdown_pct=float((curve / np.maximum.accumulate(np.r_[1, curve])[1:] - 1).min() * 100),
                   fees_fraction=sum(trade["fees"] for trade in trades),
                   mean_trade_net_return=float(np.mean([trade["net_return"] for trade in trades])) if trades else 0.0)
    metrics = {name: value.item() if isinstance(value, np.generic) else value for name, value in metrics.items()}
    return dict(metrics=metrics, trades=trades, daily_returns=daily_returns,
                equity=curve, bar_returns=returns)


def block_lower_mean(values, seed=42, repeats=500, block=7):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite daily returns required")
    random = np.random.default_rng(seed)
    starts = random.integers(0, len(values), size=(repeats, int(np.ceil(len(values) / block))))
    indices = (starts[:, :, None] + np.arange(block)) % len(values)
    samples = values[indices.reshape(repeats, -1)[:, :len(values)]]
    return float(np.quantile(samples.mean(1), 0.05))


def select_policy(data, entries, scores, min_trades=20):
    if min_trades < 1:
        raise ValueError("positive minimum trade support required")
    candidates = []
    for pattern_filter in (False, True):
        allowed = data.eligible_sides(entries) if pattern_filter else None
        for threshold in (0.0, 0.0005, 0.001, 0.002, 0.004):
            actions = choose_actions(scores, threshold, allowed)
            result = simulate(data, entries, actions)
            metrics = result["metrics"]
            daily = result["daily_returns"]
            halves_positive = bool(daily[:len(daily) // 2].sum() > 0 and daily[len(daily) // 2:].sum() > 0)
            lower_mean = block_lower_mean(daily)
            eligible = bool(metrics["trades"] >= min_trades and metrics["signals"] <= 0.5 * len(entries)
                            and metrics["total_return_pct"] > 0 and lower_mean > 0 and halves_positive)
            candidates.append(dict(threshold=threshold, pattern_filter=pattern_filter, metrics=metrics,
                                   block_lower_daily_mean=lower_mean, halves_positive=halves_positive,
                                   eligible=eligible))
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    best = max(eligible, key=lambda candidate: (candidate["block_lower_daily_mean"],
                                               -candidate["metrics"]["trades"])) if eligible else None
    return dict(enabled=best is not None, threshold=best["threshold"] if best else 0.0,
                pattern_filter=best["pattern_filter"] if best else False, min_trades=min_trades,
                candidates=candidates, execution=asdict(data.config), production_approved=False,
                note="Calibration screening, not independent significance or production approval")


def policy_actions(data, entries, scores, policy):
    if not policy["enabled"]:
        return np.zeros(len(entries), dtype=int)
    allowed = data.eligible_sides(entries) if policy["pattern_filter"] else None
    return choose_actions(scores, policy["threshold"], allowed)
