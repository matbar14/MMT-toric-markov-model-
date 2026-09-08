"""Intraday spot signals and like-for-like, long-only Toric entry filtering."""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data.payoff import causal_context, execution_outcomes, validate_market
from .execution import ExecutionConfig
from .model.payoff import ToricPayoffModel
from .train.payoff import block_lower_mean, simulate, tabular_context


SIGNALS = ("momentum_6h", "breakout_6h", "reversion_12h")
WARMUP = 100
ENTRY_HOURS_UTC = (0, 6, 12, 18)
INTRADAY_EXECUTION = ExecutionConfig(horizon=6, stop_loss=0.02, take_profit=0.03, cooldown=0)


@dataclass
class SignalData:
    frame: pd.DataFrame
    features: np.ndarray
    feature_names: list
    signals: dict
    net_returns: np.ndarray
    exit_indices: np.ndarray
    config: ExecutionConfig

    @classmethod
    def from_frame(cls, frame):
        frame = validate_market(frame)
        if len(frame) <= WARMUP + INTRADAY_EXECUTION.horizon:
            raise ValueError("more than 100 warmup hours and one complete intraday horizon required")
        if ((frame.timestamp.diff().dropna() != pd.Timedelta(hours=1)).any() or
                not frame.timestamp.equals(frame.timestamp.dt.floor("h"))):
            raise ValueError("intraday protocol requires hour-aligned UTC bars")
        features, names, _ = causal_context(frame)
        close = frame.spot_close
        extra = pd.DataFrame(dict(
            return_6h=close.pct_change(6),
            distance_to_high_6h=close / frame.spot_high.rolling(6).max().shift(1) - 1,
            zscore_12h=(close - close.rolling(12).mean()) / close.rolling(12).std().replace(0, np.nan),
            volatility_12h=close.pct_change().rolling(12).std(),
        ))
        signals = dict(momentum_6h=(extra.return_6h > 0).to_numpy(dtype=int),
                       breakout_6h=(extra.distance_to_high_6h > 0).to_numpy(dtype=int),
                       reversion_12h=(extra.zscore_12h < -1).to_numpy(dtype=int))
        for values in signals.values():
            values[:WARMUP] = 0
        features = np.column_stack((features, extra.fillna(0).to_numpy(dtype=np.float32))).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("nonfinite signal features")
        returns, exits = execution_outcomes(frame, INTRADAY_EXECUTION)
        return cls(frame, features, names + extra.columns.tolist(), signals, returns, exits, INTRADAY_EXECUTION)

    def entries(self, start, end, seq_len):
        if not 0 <= start < end <= len(self.frame) or seq_len < 1:
            raise ValueError("invalid segment bounds or sequence length")
        candidates = np.arange(max(start, WARMUP) + seq_len, end - self.config.horizon + 1, dtype=np.int64)
        if not len(candidates):
            raise ValueError("segment too short for context and complete holding horizons")
        times = self.frame.timestamp.iloc[candidates]
        valid = times.dt.hour.isin(ENTRY_HOURS_UTC)
        indices = candidates[valid.to_numpy()]
        if len(indices) < 2:
            raise ValueError("insufficient intraday entries in segment")
        return indices

    def actions(self, name, entries):
        return self.signals[name][entries - 1]

    def windows(self, entries, seq_len):
        entries = np.asarray(entries)
        if (not isinstance(seq_len, int) or seq_len < 1 or entries.ndim != 1 or not len(entries) or
                not np.issubdtype(entries.dtype, np.integer) or entries.min() < seq_len or
                entries.max() >= len(self.frame) or (np.diff(entries) <= 0).any()):
            raise ValueError("ordered entries with complete causal windows required")
        return self.features[entries[:, None] - np.arange(seq_len, 0, -1)]


def run_signal(data, entries, actions, stress=False):
    entries = np.asarray(entries)
    actions = np.asarray(actions)
    if actions.shape != np.asarray(entries).shape or not np.isin(actions, [0, 1]).all():
        raise ValueError("only aligned spot long/flat actions are supported")
    config = data.config
    if (entries.ndim != 1 or not np.issubdtype(entries.dtype, np.integer) or not len(entries) or
            (np.diff(entries) < config.horizon).any() or entries[0] < 0 or
            entries[-1] + config.horizon > len(data.frame)):
        raise ValueError("complete nonoverlapping intraday entry slots required")
    entry_times = data.frame.timestamp.iloc[entries]
    exit_times = data.frame.timestamp.iloc[entries + config.horizon - 1]
    if (not entry_times.dt.hour.isin(ENTRY_HOURS_UTC).all() or
            not np.array_equal(entry_times.dt.normalize().to_numpy(), exit_times.dt.normalize().to_numpy())):
        raise ValueError("intraday positions must close within the same UTC day")
    if stress:
        config = replace(config, fee=config.fee * 1.5, slippage=config.slippage * 2)
    result = simulate(data, entries, actions, config)
    result["daily_dates"] = [date.isoformat() for date in pd.date_range(
        entry_times.iloc[0].normalize(), exit_times.iloc[-1].normalize(), freq="D")]
    result["metrics"]["block_lower_daily_mean"] = block_lower_mean(result["daily_returns"])
    if result["metrics"]["signals"] != result["metrics"]["trades"]:
        raise ValueError("intraday comparison must not silently skip scheduled entries")
    result["metrics"]["trades_per_day"] = result["metrics"]["trades"] / len(result["daily_returns"])
    return result


def select_signal(metrics, min_trades=30):
    if min_trades < 1 or set(metrics) != set(SIGNALS):
        raise ValueError("complete fixed candidate set and positive minimum trades required")
    eligible = [name for name in SIGNALS if metrics[name]["trades"] >= min_trades and
                metrics[name]["trades_per_day"] >= 0.5]
    return max(eligible, key=lambda name: (metrics[name]["block_lower_daily_mean"],
                                         metrics[name]["total_return_pct"], -SIGNALS.index(name))) if eligible else None


def base_screen(run, stress, min_trades=30):
    metrics = run["metrics"]
    return bool(metrics["trades"] >= min_trades and metrics["trades_per_day"] >= 0.5 and
                metrics["total_return_pct"] > 0 and
                stress["metrics"]["total_return_pct"] > 0 and metrics["block_lower_daily_mean"] > 0)


def filter_actions(base_actions, expected_returns):
    base_actions, expected_returns = np.asarray(base_actions), np.asarray(expected_returns)
    if (base_actions.shape != expected_returns.shape or base_actions.ndim != 1 or
            not np.isin(base_actions, [0, 1]).all() or not np.isfinite(expected_returns).all()):
        raise ValueError("aligned finite forecasts and long/flat base actions required")
    return (base_actions * (expected_returns > 0)).astype(int)


def paired_comparison(base, filtered, base_stress, filtered_stress, min_trades=30):
    if (base.get("daily_dates") is None or
            any(run.get("daily_dates") != base["daily_dates"] or
                len(run["daily_returns"]) != len(base["daily_dates"])
                for run in (base, filtered, base_stress, filtered_stress))):
        raise ValueError("paired comparison requires identical daily evaluation dates")
    difference = filtered["daily_returns"] - base["daily_returns"]
    metrics, baseline = filtered["metrics"], base["metrics"]
    lower = block_lower_mean(difference)
    retained = metrics["trades"] / max(baseline["trades"], 1)
    improved = bool(metrics["trades"] >= min_trades and metrics["trades_per_day"] >= 0.5 and
                    retained >= 0.5 and lower > 0 and
                    metrics["total_return_pct"] > max(0, baseline["total_return_pct"]) and
                    metrics["max_drawdown_pct"] >= baseline["max_drawdown_pct"] and
                    filtered_stress["metrics"]["total_return_pct"] > max(0, base_stress["metrics"]["total_return_pct"]))
    return dict(improved=improved, retained_fraction=retained, paired_lower_daily_mean=lower,
                return_difference_pct=metrics["total_return_pct"] - baseline["total_return_pct"],
                stress_difference_pct=filtered_stress["metrics"]["total_return_pct"] - base_stress["metrics"]["total_return_pct"],
                note="Developmental screening only; no multiple-testing or independent-holdout guarantee")


def train_entry_filters(data, train_entries, selection_entries, seq_len=32, epochs=12, seeds=(42, 43, 44)):
    if epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("positive epochs and distinct fixed ensemble seeds required")
    torch.set_num_threads(1)
    train_windows = data.windows(train_entries, seq_len)
    selected_windows = torch.from_numpy(data.windows(selection_entries, seq_len))
    if train_entries[-1] + data.config.horizon > selection_entries[0] - seq_len:
        raise ValueError("training targets and selection context must be chronologically disjoint")
    targets = torch.tensor(data.net_returns[train_entries, :1], dtype=torch.float32)
    validation_targets = torch.tensor(data.net_returns[selection_entries, :1], dtype=torch.float32)
    if len(targets) < 20:
        raise ValueError("fewer than 20 intraday training targets; refusing to fit neural filter")
    models, histories = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = ToricPayoffModel(data.features.shape[1], max_len=seq_len, dim_angles=16,
                                num_states=8, dropout=0.1, num_outputs=1)
        model.fit_statistics(data.features[train_entries[0] - seq_len:train_entries[-1]], targets)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        loader = DataLoader(TensorDataset(torch.from_numpy(train_windows), targets), batch_size=32,
                            shuffle=True, generator=torch.Generator().manual_seed(seed))
        best_loss, best_state, best_epoch, stale = float("inf"), None, None, 0
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            for features, target in loader:
                optimizer.zero_grad()
                loss = (model(features) - (target - model.target_mean) / model.target_std).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite filter loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                forecast = torch.cat([model.predict_payoffs(batch) for batch in selected_windows.split(32)])
                value = (forecast - validation_targets).square().mean().item()
            if not np.isfinite(value):
                raise ValueError("nonfinite filter selection error")
            history.append(dict(epoch=epoch, selection_mse=value))
            if value < best_loss - 1e-9:
                best_loss, best_state, best_epoch, stale = value, deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= 4:
                break
        model.load_state_dict(best_state)
        models.append(model.eval())
        histories.append(dict(seed=seed, best_epoch=best_epoch, selection_mse=best_loss, history=history))
        print(f"filter seed={seed}, selected epoch={best_epoch}, selection MSE={best_loss:.6g}", flush=True)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100))
    ridge.fit(tabular_context(train_windows), targets.numpy().reshape(-1))
    return models, ridge, histories


def predict_filters(models, ridge, windows):
    if not models or any(model.training for model in models):
        raise ValueError("nonempty evaluation-mode ensemble required")
    members = []
    with torch.inference_mode():
        for model in models:
            members.append(torch.cat([model.predict_payoffs(batch) for batch in torch.from_numpy(windows).split(32)]).numpy().reshape(-1))
    return dict(toric=np.mean(members, axis=0), ridge=ridge.predict(tabular_context(windows))), members


def matched_random_control(data, entries, base_actions, filtered_actions, seed=42, repeats=100):
    if repeats < 1:
        raise ValueError("positive random-control sample count required")
    base_actions, filtered_actions = np.asarray(base_actions), np.asarray(filtered_actions)
    if (base_actions.shape != filtered_actions.shape or not np.isin(base_actions, [0, 1]).all() or
            not np.isin(filtered_actions, [0, 1]).all() or (filtered_actions > base_actions).any()):
        raise ValueError("matched control requires a long/flat subset of base opportunities")
    eligible = np.flatnonzero(base_actions)
    retained = int(filtered_actions.sum())
    random = np.random.default_rng(seed)
    results = []
    for _ in range(repeats):
        actions = np.zeros(len(entries), dtype=int)
        actions[random.choice(eligible, size=retained, replace=False)] = 1
        results.append(run_signal(data, entries, actions)["metrics"]["total_return_pct"])
    actual = run_signal(data, entries, filtered_actions)["metrics"]["total_return_pct"]
    return dict(retained_trades=retained, trials=repeats, median_return_pct=float(np.median(results)),
                quantiles_return_pct=np.quantile(results, [0.05, 0.5, 0.95]).tolist(),
                filter_return_pct=actual, fraction_random_at_least_as_good=float(np.mean(np.array(results) >= actual)),
                informative=bool(0 < retained < len(eligible)),
                note="Conditional count-matched historical control, not a calibrated significance test")
