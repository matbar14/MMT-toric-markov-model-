"""Causal market context and execution-consistent long/short payoff targets."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..execution import ExecutionConfig, exit_on_bar


PATTERN_NAMES = (
    "bullish_div", "bearish_div", "cvd_reversal_bull", "cvd_reversal_bear",
    "cvd_exhaustion_bull", "cvd_exhaustion_bear", "cvd_spot_futures_bull", "cvd_spot_futures_bear",
    "cvd_spike_bull", "cvd_spike_bear", "basis_long", "basis_short",
)


def validate_market(frame):
    required = [f"{market}_{field}" for market in ("spot", "futures")
                for field in ("open", "high", "low", "close", "volume", "cvd")]
    if "timestamp" not in frame or any(name not in frame for name in required):
        raise ValueError("timestamp and spot/futures OHLCV/CVD columns required")
    if len(frame) < 2:
        raise ValueError("at least two market rows required")
    frame = frame.copy().reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    timestamps = frame.timestamp
    if (timestamps.isna().any() or not timestamps.is_monotonic_increasing or
            timestamps.duplicated().any() or timestamps.diff().dropna().nunique() != 1):
        raise ValueError("timestamps must be ordered, unique and regularly spaced")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("market values must be finite")
    for market in ("spot", "futures"):
        prices = frame[[f"{market}_{field}" for field in ("open", "high", "low", "close")]]
        if ((prices <= 0).any().any() or (frame[f"{market}_volume"] <= 0).any() or
                (prices[f"{market}_high"] < prices.max(axis=1)).any() or
                (prices[f"{market}_low"] > prices.min(axis=1)).any()):
            raise ValueError("invalid OHLC or volume")
    return frame


def causal_context(frame):
    """Return unnormalized features available at each bar CLOSE; no targets read."""
    close = frame.spot_close
    spot_delta = frame.spot_cvd.diff()
    futures_delta = frame.futures_cvd.diff()
    basis = frame.futures_close - close
    basis_pct = basis / close
    basis_z = (basis - basis.rolling(100).mean()) / (basis.rolling(100).std() + 1e-8)
    price_short = close.diff(3)
    price_long = close.diff(14)
    momentum = frame.spot_cvd.diff(2)
    cvd_short = frame.spot_cvd.diff(3)
    futures_short = frame.futures_cvd.diff(3)
    cvd_long = frame.spot_cvd.diff(14)
    move = close * (close.pct_change().rolling(14).std() * 1.5).clip(0.002, 0.03)
    cvd_z = (frame.spot_cvd - frame.spot_cvd.rolling(100).mean()) / (frame.spot_cvd.rolling(100).std() + 1e-8)
    volume_spike = frame.spot_volume > frame.spot_volume.rolling(20).mean() * 1.5
    patterns = np.column_stack((
        (price_long < -move) & (cvd_long > 0) & (price_short > 0),
        (price_long > move) & (cvd_long < 0) & (price_short < 0),
        (momentum.shift(3) < 0) & (momentum > 0) & (price_short >= 0),
        (momentum.shift(3) > 0) & (momentum < 0) & (price_short <= 0),
        (cvd_z < -2.5) & (momentum > 0) & (price_short > 0),
        (cvd_z > 2.5) & (momentum < 0) & (price_short < 0),
        (cvd_short > 0) & (futures_short < 0) & (price_short >= 0),
        (cvd_short < 0) & (futures_short > 0) & (price_short <= 0),
        (momentum > momentum.rolling(20).mean() + 2 * momentum.rolling(20).std()) & volume_spike & (price_short > 0),
        (momentum < momentum.rolling(20).mean() - 2 * momentum.rolling(20).std()) & volume_spike & (price_short < 0),
        (basis_z < -2) & (basis.diff(2) > 0),
        (basis_z > 2) & (basis.diff(2) < 0),
    )).astype(bool)
    columns = {}
    for window in (1, 3, 6, 12, 24):
        columns[f"spot_return_{window}"] = close.pct_change(window)
        columns[f"futures_return_{window}"] = frame.futures_close.pct_change(window)
        columns[f"spot_order_flow_{window}"] = spot_delta.rolling(window).sum() / frame.spot_volume.rolling(window).sum()
        columns[f"futures_order_flow_{window}"] = futures_delta.rolling(window).sum() / frame.futures_volume.rolling(window).sum()
    for window in (6, 24, 100):
        columns[f"volatility_{window}"] = close.pct_change().rolling(window).std()
        columns[f"volume_relative_{window}"] = frame.spot_volume / frame.spot_volume.rolling(window).mean() - 1
    columns.update(
        candle_range=(frame.spot_high - frame.spot_low) / close,
        candle_body=(close - frame.spot_open) / frame.spot_open,
        basis_pct=basis_pct, basis_pct_change=basis_pct.diff(), basis_zscore=basis_z,
        volume_ratio=frame.futures_volume / frame.spot_volume,
    )
    columns.update({f"candidate_{name}": patterns[:, index].astype(float)
                    for index, name in enumerate(PATTERN_NAMES)})
    features = pd.DataFrame(columns).to_numpy(dtype=np.float32)
    if not np.isfinite(features[100:]).all():
        raise ValueError("nonfinite features after warmup")
    return np.nan_to_num(features[:], nan=0.0), list(columns), patterns


def execution_outcomes(frame, config):
    """Counterfactual trades per entry; [long, short], including round-trip costs."""
    rows = len(frame)
    net_returns = np.full((rows, 2), np.nan)
    exit_indices = np.full((rows, 2), -1, dtype=np.int64)
    prices = frame[["spot_open", "spot_high", "spot_low", "spot_close"]].to_numpy()
    for entry in range(rows - config.horizon + 1):
        for column, side in enumerate((1, -1)):
            entry_fill = prices[entry, 0] * (1 + side * config.slippage)
            for exit_index in range(entry, entry + config.horizon):
                bar_open, bar_high, bar_low, bar_close = prices[exit_index]
                exited, raw_exit, _ = exit_on_bar(
                    side, entry_fill, bar_open, bar_high, bar_low, bar_close,
                    config.stop_loss, config.take_profit, exit_index - entry + 1, config.horizon,
                )
                if exited:
                    exit_fill = raw_exit * (1 - side * config.slippage)
                    ratio = exit_fill / entry_fill
                    net_returns[entry, column] = side * (ratio - 1) - config.fee * (1 + ratio)
                    exit_indices[entry, column] = exit_index
                    break
    return net_returns, exit_indices


@dataclass
class PayoffData:
    frame: pd.DataFrame
    features: np.ndarray
    feature_names: list
    candidates: np.ndarray
    net_returns: np.ndarray
    exit_indices: np.ndarray
    config: ExecutionConfig

    @classmethod
    def from_frame(cls, frame, config=ExecutionConfig()):
        frame = validate_market(frame)
        features, names, patterns = causal_context(frame)
        returns, exits = execution_outcomes(frame, config)
        return cls(frame, features, names, patterns, returns, exits, config)

    def entries(self, start, end, seq_len):
        first = max(start, 100) + seq_len
        indices = np.arange(first, end - self.config.horizon + 1, dtype=np.int64)
        if start < 0 or end > len(self.frame) or seq_len < 1 or not len(indices):
            raise ValueError("segment too short for disjoint context and complete outcomes")
        return indices

    def windows(self, entries, seq_len):
        return self.features[entries[:, None] - np.arange(seq_len, 0, -1)]

    def eligible_sides(self, entries):
        patterns = self.candidates[entries - 1]
        return np.column_stack((patterns[:, ::2].any(1), patterns[:, 1::2].any(1)))


def walk_forward_segments(rows, folds=3):
    if folds < 2 or rows < 200:
        raise ValueError("at least two folds and sufficient history required")
    validation_rows = int(rows * 0.1)
    first_evaluation = int(rows * 0.6)
    boundaries = np.linspace(first_evaluation, rows, folds + 1, dtype=int)
    return [dict(train=(0, int(start - 2 * validation_rows)),
                 selection=(int(start - 2 * validation_rows), int(start - validation_rows)),
                 calibration=(int(start - validation_rows), int(start)),
                 evaluation=(int(start), int(end)))
            for start, end in zip(boundaries[:-1], boundaries[1:])]
