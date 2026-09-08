"""Research-only BTC spot/linear USDT perpetual paired execution and causal inputs."""

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd

from .data.payoff import validate_market
from .train.payoff import block_lower_mean


ENTRY_HOURS_UTC = (0, 7, 14)
WARMUP = 48


@dataclass(frozen=True)
class ArbitrageConfig:
    horizon: int = 6
    spot_fee: float = 0.001
    futures_fee: float = 0.0005
    spot_slippage: float = 0.0002
    futures_slippage: float = 0.0002
    allocation: float = 0.8
    margin_ratio: float = 1.0
    margin_buffer: float = 0.02
    min_edge: float = 0.0005
    max_drawdown: float = 0.05
    funding_guard_seconds: int = 15

    def __post_init__(self):
        if type(self.horizon) is not int or not 1 <= self.horizon <= 6:
            raise ValueError("intraday horizon must be an integer from one to six hours")
        if type(self.funding_guard_seconds) is not int or not 0 <= self.funding_guard_seconds <= 60:
            raise ValueError("funding boundary guard must be 0..60 seconds")
        for name in ("spot_fee", "futures_fee", "spot_slippage", "futures_slippage", "allocation", "margin_buffer", "min_edge", "max_drawdown"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"invalid {name}")
        if (min(self.allocation, self.margin_buffer, self.max_drawdown) <= 0 or
                not math.isfinite(self.margin_ratio) or self.margin_ratio < 1):
            raise ValueError("positive allocation/buffer and at least 100% initial short collateral required")

    def stressed(self):
        return replace(self, spot_fee=self.spot_fee * 1.5, futures_fee=self.futures_fee * 1.5,
                       spot_slippage=self.spot_slippage * 2, futures_slippage=self.futures_slippage * 2)


def validate_funding(frame):
    required = ["timestamp", "funding_rate", "mark_price"]
    if any(name not in frame for name in required) or len(frame) < 2:
        raise ValueError("settled funding history with timestamp, funding_rate and mark_price required")
    result = frame[required].copy().reset_index(drop=True)
    result.timestamp = pd.to_datetime(result.timestamp, utc=True, errors="raise", format="mixed").astype("datetime64[ns, UTC]")
    values = result[["funding_rate", "mark_price"]].apply(pd.to_numeric, errors="raise")
    if (result.timestamp.isna().any() or result.timestamp.duplicated().any() or
            not result.timestamp.is_monotonic_increasing or not np.isfinite(values.to_numpy()).all() or
            (values.mark_price <= 0).any() or (values.funding_rate.abs() >= 1).any()):
        raise ValueError("invalid or duplicate funding events")
    result[["funding_rate", "mark_price"]] = values
    if (result.timestamp.diff().dropna() > pd.Timedelta(hours=9)).any():
        raise ValueError("funding coverage gap over nine hours; do not interpolate settlements")
    return result


def validate_marks(frame, timestamps):
    required = ["timestamp", "open", "high", "low", "close"]
    if any(name not in frame for name in required):
        raise ValueError("mark-price OHLC history required; trade prices cannot replace mark prices")
    result = frame[required].copy().reset_index(drop=True)
    result.timestamp = pd.to_datetime(result.timestamp, utc=True, errors="raise").astype("datetime64[ns, UTC]")
    prices = result[required[1:]].apply(pd.to_numeric, errors="raise")
    if (not result.timestamp.equals(timestamps.reset_index(drop=True).astype("datetime64[ns, UTC]")) or
            not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any() or
            (prices.high < prices.max(axis=1)).any() or (prices.low > prices.min(axis=1)).any()):
        raise ValueError("mark-price OHLC must be valid and exactly aligned with market candles")
    result[required[1:]] = prices
    return result


@dataclass
class ArbitrageData:
    frame: pd.DataFrame
    funding: pd.DataFrame
    marks: pd.DataFrame
    features: np.ndarray
    feature_names: list
    config: ArbitrageConfig

    @classmethod
    def from_frames(cls, frame, funding, marks, config=None):
        config = config or ArbitrageConfig()
        frame = validate_market(frame)
        frame.timestamp = frame.timestamp.astype("datetime64[ns, UTC]")
        if ((frame.timestamp.diff().dropna() != pd.Timedelta(hours=1)).any() or
                not frame.timestamp.equals(frame.timestamp.dt.floor("h"))):
            raise ValueError("hour-aligned UTC market bars required")
        funding = validate_funding(funding)
        end = frame.timestamp.iloc[-1] + pd.Timedelta(hours=1, seconds=config.funding_guard_seconds)
        start = frame.timestamp.iloc[0] - pd.Timedelta(seconds=config.funding_guard_seconds)
        if funding.timestamp.iloc[0] >= start or funding.timestamp.iloc[-1] <= end:
            raise ValueError("funding must bracket the whole market window, including boundary guards")
        marks = validate_marks(marks, frame.timestamp)
        known = pd.merge_asof(
            pd.DataFrame({"known_at": frame.timestamp + pd.Timedelta(hours=1) -
                          pd.Timedelta(seconds=config.funding_guard_seconds)}),
            funding.assign(interval_hours=funding.timestamp.diff().dt.total_seconds() / 3600),
            left_on="known_at", right_on="timestamp", direction="backward", allow_exact_matches=False)
        basis = frame.futures_close / frame.spot_close - 1
        features = pd.DataFrame(dict(
            basis=basis, basis_change_1h=basis.diff(), basis_change_6h=basis.diff(6),
            basis_z_24h=(basis - basis.rolling(24).mean()) / basis.rolling(24).std().replace(0, np.nan),
            basis_vol_24h=basis.diff().rolling(24).std(), settled_funding=known.funding_rate,
            funding_age_hours=(known.known_at - known.timestamp).dt.total_seconds() / 3600,
            settled_interval_hours=known.interval_hours,
            mark_premium=marks.close / frame.spot_close - 1,
            mark_trade_gap=marks.close / frame.futures_close - 1,
            spot_return_1h=frame.spot_close.pct_change(), spot_return_6h=frame.spot_close.pct_change(6),
            volume_ratio=np.log(frame.futures_volume / frame.spot_volume),
        )).fillna(0)
        values = features.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite arbitrage features")
        return cls(frame, funding, marks, values, features.columns.tolist(), config)

    def entries(self, start, end, seq_len=32):
        if not 0 <= start < end <= len(self.frame) or type(seq_len) is not int or seq_len < 1:
            raise ValueError("invalid segment or context length")
        indices = np.arange(max(start, WARMUP) + seq_len, end - self.config.horizon + 1, dtype=np.int64)
        indices = indices[self.frame.timestamp.iloc[indices].dt.hour.isin(ENTRY_HOURS_UTC)]
        if not len(indices):
            raise ValueError("segment has no complete intraday opportunities")
        return indices

    def windows(self, entries, seq_len=32):
        entries = np.asarray(entries)
        if (entries.ndim != 1 or not len(entries) or not np.issubdtype(entries.dtype, np.integer) or
                type(seq_len) is not int or seq_len < 1 or entries.min() < seq_len or
                entries.max() >= len(self.frame) or (np.diff(entries) <= 0).any()):
            raise ValueError("ordered complete causal windows required")
        return self.features[entries[:, None] - np.arange(seq_len, 0, -1)]

    def baseline_actions(self, entries):
        context = self.windows(entries, 1)[:, 0]
        spot_in, spot_out = 1 + self.config.spot_slippage, 1 - self.config.spot_slippage
        futures_in = (1 + context[:, 0]) * (1 - self.config.futures_slippage)
        futures_out = 1 + self.config.futures_slippage
        costs = (spot_in + spot_out) * self.config.spot_fee + (futures_in + futures_out) * self.config.futures_fee
        committed = spot_in * (1 + self.config.spot_fee) + futures_in * self.config.margin_ratio
        hypothetical_return = (spot_out - spot_in + futures_in - futures_out - costs) / committed
        return ((hypothetical_return > self.config.min_edge) &
                (context[:, self.feature_names.index("settled_funding")] >= 0)).astype(int)

    def carry_actions(self, entries):
        context = self.windows(entries, 1)[:, 0]
        spot_in, spot_out = 1 + self.config.spot_slippage, 1 - self.config.spot_slippage
        futures_in = (1 + context[:, 0]) * (1 - self.config.futures_slippage)
        futures_out = (1 + context[:, 0]) * (1 + self.config.futures_slippage)
        costs = (spot_in + spot_out) * self.config.spot_fee + (futures_in + futures_out) * self.config.futures_fee
        committed = spot_in * (1 + self.config.spot_fee) + futures_in * self.config.margin_ratio
        interval = context[:, self.feature_names.index("settled_interval_hours")]
        rate = context[:, self.feature_names.index("settled_funding")]
        mark_ratio = 1 + context[:, self.feature_names.index("mark_premium")]
        estimated_funding = rate * mark_ratio * self.config.horizon / np.maximum(interval, 1e-6)
        estimate = (spot_out - spot_in + futures_in - futures_out + estimated_funding - costs) / committed
        return ((interval > 0) & (estimate > self.config.min_edge)).astype(int)


def pair_outcome(data, entry, config=None):
    config = config or data.config
    if (not isinstance(entry, (int, np.integer)) or entry < 0 or entry + config.horizon > len(data.frame) or
            data.frame.timestamp.iloc[entry].hour not in ENTRY_HOURS_UTC):
        raise ValueError("complete scheduled pair entry required")
    exit_index = entry + config.horizon - 1
    start = data.frame.timestamp.iloc[entry]
    end = data.frame.timestamp.iloc[exit_index] + pd.Timedelta(hours=1)
    spot_in = float(data.frame.spot_open.iloc[entry]) * (1 + config.spot_slippage)
    futures_in = float(data.frame.futures_open.iloc[entry]) * (1 - config.futures_slippage)
    spot_out = float(data.frame.spot_close.iloc[exit_index]) * (1 - config.spot_slippage)
    futures_out = float(data.frame.futures_close.iloc[exit_index]) * (1 + config.futures_slippage)
    entry_fees = spot_in * config.spot_fee + futures_in * config.futures_fee
    exit_fees = spot_out * config.spot_fee + futures_out * config.futures_fee
    collateral = futures_in * config.margin_ratio
    committed = spot_in * (1 + config.spot_fee) + collateral
    if not np.isfinite([spot_in, futures_in, spot_out, futures_out, entry_fees, exit_fees, collateral, committed]).all():
        raise ValueError("nonfinite pair notional or costs")
    guard = pd.Timedelta(seconds=config.funding_guard_seconds)
    events = data.funding.loc[data.funding.timestamp.between(start - guard, end + guard)].copy()
    interior = (events.timestamp > start + guard) & (events.timestamp < end - guard)
    events["cash"] = (events.funding_rate * events.mark_price).where(interior | (events.funding_rate < 0), 0)
    funding_per_bar = np.zeros(config.horizon)
    negative_funding_per_bar = np.zeros(config.horizon)
    for event in events.itertuples():
        offset = min(max(int((event.timestamp - start).total_seconds() // 3600), 0), config.horizon - 1)
        funding_per_bar[offset] += event.cash
        negative_funding_per_bar[offset] += min(event.cash, 0)
    equity_per_unit, minimum_margin_buffer = [], float("inf")
    settled = 0.0
    for offset, bar in enumerate(range(entry, exit_index + 1)):
        mark_high = float(data.marks.high.iloc[bar])
        available = collateral - futures_in * config.futures_fee + settled + negative_funding_per_bar[offset]
        margin_equity_low = available + futures_in - mark_high
        margin_buffer = margin_equity_low - mark_high * (config.margin_buffer + config.futures_fee)
        minimum_margin_buffer = min(minimum_margin_buffer, margin_buffer)
        if margin_buffer <= 0:
            raise ValueError(f"short collateral risk at {data.frame.timestamp.iloc[bar]}; "
                             "hourly research cannot certify liquidation safety")
        settled += float(funding_per_bar[offset])
        equity_per_unit.append(float(data.frame.spot_close.iloc[bar]) - spot_in + futures_in -
                               float(data.marks.close.iloc[bar]) + settled - entry_fees)
    spot_pnl, futures_pnl = spot_out - spot_in, futures_in - futures_out
    net = spot_pnl + futures_pnl + settled - entry_fees - exit_fees
    equity_per_unit[-1] = net
    if not np.isfinite(equity_per_unit).all():
        raise ValueError("nonfinite paired PnL")
    return dict(entry=int(entry), exit=int(exit_index), entry_time=start.isoformat(), exit_time=end.isoformat(),
                spot_quantity=1.0, futures_quantity=-1.0, delta_btc=0.0,
                spot_entry=spot_in, spot_exit=spot_out, futures_entry=futures_in, futures_exit=futures_out,
                committed_per_unit=committed, spot_pnl=spot_pnl, futures_pnl=futures_pnl,
                funding_pnl=float(settled), fees=entry_fees + exit_fees, net_pnl=net,
                net_return=net / committed, min_margin_buffer=minimum_margin_buffer,
                boundary_events=int((~interior).sum()), equity_per_unit=np.asarray(equity_per_unit))


def simulate_pairs(data, entries, actions, config=None):
    config = config or data.config
    entries, actions = np.asarray(entries), np.asarray(actions)
    if (entries.ndim != 1 or not len(entries) or not np.issubdtype(entries.dtype, np.integer) or
            actions.shape != entries.shape or not np.isin(actions, [0, 1]).all() or
            (np.diff(entries) < config.horizon + 1).any() or entries[0] < 0 or
            entries[-1] + config.horizon > len(data.frame) or
            not data.frame.timestamp.iloc[entries].dt.hour.isin(ENTRY_HOURS_UTC).all()):
        raise ValueError("nonoverlapping scheduled pairs and aligned OPEN_PAIR/FLAT actions required")
    first, last = int(entries[0]), int(entries[-1] + config.horizon)
    curve, trades, capital, cursor = np.ones(last - first), [], 1.0, first
    for entry, action in zip(entries, actions):
        if not action:
            continue
        outcome = pair_outcome(data, int(entry), config)
        quantity = capital * config.allocation / outcome["committed_per_unit"]
        curve[cursor - first:entry - first] = capital
        curve[entry - first:entry + config.horizon - first] = capital + quantity * outcome["equity_per_unit"]
        outcome.pop("equity_per_unit")
        for name in ("spot_pnl", "futures_pnl", "funding_pnl", "fees", "net_pnl", "min_margin_buffer"):
            outcome[name] *= quantity
        outcome.update(spot_quantity=quantity, futures_quantity=-quantity,
                       committed_capital=quantity * outcome["committed_per_unit"])
        capital += outcome["net_pnl"]
        cursor = int(entry + config.horizon)
        trades.append(outcome)
    curve[cursor - first:] = capital
    if not np.isfinite(curve).all() or (curve <= 0).any():
        raise ValueError("bankrupt paired portfolio")
    times = pd.DatetimeIndex(data.frame.timestamp.iloc[first:last])
    daily = pd.Series(curve, index=times).resample("1D").last()
    daily_returns = daily.pct_change().fillna(daily.iloc[0] - 1).to_numpy()
    metrics = dict(trades=len(trades), signals=int(actions.sum()), total_return_pct=(capital - 1) * 100,
                   trades_per_day=len(trades) / len(daily), fees_fraction=sum(trade["fees"] for trade in trades),
                   funding_fraction=sum(trade["funding_pnl"] for trade in trades),
                   max_drawdown_pct=float((curve / np.maximum.accumulate(np.r_[1, curve])[1:] - 1).min() * 100),
                   block_lower_daily_mean=block_lower_mean(daily_returns),
                   max_abs_delta_btc=max((abs(trade["spot_quantity"] + trade["futures_quantity"])
                                         for trade in trades), default=0.0))
    return dict(metrics=metrics, trades=trades, equity=curve, daily_returns=daily_returns,
                daily_dates=[stamp.isoformat() for stamp in daily.index])
