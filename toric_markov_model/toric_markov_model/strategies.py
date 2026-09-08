"""Predeclared causal rule strategies and frozen-at-entry volatility exits."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .data.payoff import causal_context, validate_market
from .execution import ExecutionConfig
from .train.payoff import block_lower_mean, simulate


FAMILIES = ("cvd_rules", "trend_breakout", "trend_pullback", "range_reversion")
EXIT_MODES = ("fixed_1pct", "fixed_2pct", "atr_2", "atr_3")
HORIZONS = (4, 12, 24)


def wilder_atr(frame, period=14):
    if not isinstance(period, int) or period < 1 or len(frame) < period:
        raise ValueError("positive ATR period and sufficient rows required")
    previous_close = frame.spot_close.shift(1)
    true_range = pd.concat((frame.spot_high - frame.spot_low,
                            (frame.spot_high - previous_close).abs(),
                            (frame.spot_low - previous_close).abs()), axis=1).max(axis=1).to_numpy()
    values = np.full(len(frame), np.nan)
    values[period - 1] = true_range[:period].mean()
    for index in range(period, len(frame)):
        values[index] = (values[index - 1] * (period - 1) + true_range[index]) / period
    return pd.Series(values, index=frame.index)


def strategy_indicators(frame):
    close = frame.spot_close
    atr = wilder_atr(frame)
    indicators = pd.DataFrame(dict(atr=atr, atr_pct=atr / close))
    for period in (20, 50, 200):
        indicators[f"ema_{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()
    indicators["channel_high"] = frame.spot_high.rolling(24).max().shift(1)
    indicators["channel_low"] = frame.spot_low.rolling(24).min().shift(1)
    indicators["relative_volume"] = frame.spot_volume / frame.spot_volume.rolling(24).mean()
    indicators["range_zscore"] = (close - close.rolling(48).mean()) / close.rolling(48).std().replace(0, np.nan)
    return indicators


def strategy_signals(frame, indicators):
    close = frame.spot_close
    ema20, ema50, ema200 = (indicators[f"ema_{period}"] for period in (20, 50, 200))
    up = (ema50 > ema200) & (close > ema200)
    down = (ema50 < ema200) & (close < ema200)
    volume = indicators.relative_volume > 1
    upper, lower = indicators.channel_high, indicators.channel_low
    breakout_long = up & volume & (close > upper) & (close.shift(1) <= upper.shift(1))
    breakout_short = down & volume & (close < lower) & (close.shift(1) >= lower.shift(1))
    pullback_long = up & (close > ema20) & (close.shift(1) <= ema20.shift(1))
    pullback_short = down & (close < ema20) & (close.shift(1) >= ema20.shift(1))
    range_regime = (ema50 - ema200).abs() < 2 * indicators.atr
    zscore = indicators.range_zscore
    reversion_long = range_regime & (zscore.shift(1) < -2) & (zscore >= -2)
    reversion_short = range_regime & (zscore.shift(1) > 2) & (zscore <= 2)
    _, _, patterns = causal_context(frame)
    signals = dict(cvd_rules=np.sign(patterns[:, ::2].sum(1) - patterns[:, 1::2].sum(1)).astype(int))
    for name, long_signal, short_signal in (
        ("trend_breakout", breakout_long, breakout_short),
        ("trend_pullback", pullback_long, pullback_short),
        ("range_reversion", reversion_long, reversion_short),
    ):
        signals[name] = long_signal.to_numpy(dtype=int) - short_signal.to_numpy(dtype=int)
    for values in signals.values():
        values[:200] = 0
    return signals


@dataclass(frozen=True)
class StrategySpec:
    family: str
    exit_mode: str
    horizon: int

    def __post_init__(self):
        if self.family not in FAMILIES or self.exit_mode not in EXIT_MODES or self.horizon not in HORIZONS:
            raise ValueError("strategy outside the predeclared experiment")

    @property
    def name(self):
        return f"{self.family}__{self.exit_mode}__h{self.horizon}"


@dataclass
class StrategyData:
    frame: pd.DataFrame
    indicators: pd.DataFrame
    signals: dict
    config: ExecutionConfig

    @classmethod
    def from_frame(cls, frame, config=ExecutionConfig()):
        frame = validate_market(frame)
        indicators = strategy_indicators(frame)
        return cls(frame, indicators, strategy_signals(frame, indicators), config)

    def entries(self, start, end):
        if not 0 <= start < end <= len(self.frame):
            raise ValueError("invalid segment boundaries")
        entries = np.arange(max(start + 1, 201), end - max(HORIZONS) + 1, dtype=np.int64)
        if not len(entries):
            raise ValueError("segment too short for common entry range and complete horizons")
        return entries

    def distances(self, entries, mode):
        if mode not in EXIT_MODES:
            raise ValueError("unknown exit mode")
        if mode.startswith("fixed"):
            stop = 0.01 if mode == "fixed_1pct" else 0.02
            return np.full(len(entries), stop), np.full(len(entries), 0.02)
        atr_pct = self.indicators.atr_pct.to_numpy()[entries - 1]
        multiple = 2 if mode == "atr_2" else 3
        return multiple * atr_pct, 4 * atr_pct


def strategy_grid():
    return [StrategySpec(family, mode, horizon) for family in FAMILIES
            for mode in EXIT_MODES for horizon in HORIZONS]


def evaluate_strategy(data, entries, spec, risk_fraction=0.002, stress=False, actions=None):
    config = replace(data.config, horizon=spec.horizon)
    if stress:
        config = replace(config, fee=config.fee * 1.5, slippage=config.slippage * 2)
    stop, take = data.distances(entries, spec.exit_mode)
    actions = data.signals[spec.family][entries - 1] if actions is None else actions
    result = simulate(data, entries, actions, config, stop_distances=stop, take_distances=take,
                      risk_fraction=risk_fraction)
    padding = max(HORIZONS) - spec.horizon
    if padding:
        result["equity"] = np.r_[result["equity"], np.repeat(result["equity"][-1], padding)]
        result["bar_returns"] = np.r_[result["bar_returns"], np.zeros(padding)]
        times = data.frame.timestamp.iloc[entries[0]:entries[-1] + max(HORIZONS)]
        daily_equity = pd.Series(result["equity"], index=pd.DatetimeIndex(times)).resample("1D").last()
        result["daily_returns"] = daily_equity.pct_change().fillna(daily_equity.iloc[0] - 1).to_numpy()
    daily = result["daily_returns"]
    result["metrics"].update(
        block_lower_daily_mean=block_lower_mean(daily),
        positive_calendar_halves=bool(daily[:len(daily) // 2].sum() > 0 and daily[len(daily) // 2:].sum() > 0),
        stop_exits=sum(trade["reason"] == "STOP_LOSS" for trade in result["trades"]),
        take_exits=sum(trade["reason"] == "TAKE_PROFIT" for trade in result["trades"]),
        time_exits=sum(trade["reason"] == "MAX_HOLD" for trade in result["trades"]),
        mean_holding_bars=float(np.mean([trade["exit"] - trade["entry"] + 1 for trade in result["trades"]]))
        if result["trades"] else 0.0,
    )
    return result


def choose_on_selection(metrics, min_trades=10):
    if min_trades < 1:
        raise ValueError("positive trade support required")
    eligible = [name for name, values in metrics.items() if values["trades"] >= min_trades]
    return max(eligible, key=lambda name: (metrics[name]["block_lower_daily_mean"],
                                         metrics[name]["total_return_pct"], -metrics[name]["trades"], name)) if eligible else None


def calibration_passes(metrics, stress_metrics, min_trades=10):
    return bool(metrics["trades"] >= min_trades and metrics["total_return_pct"] > 0 and
                metrics["positive_calendar_halves"] and metrics["block_lower_daily_mean"] > 0 and
                stress_metrics["total_return_pct"] > 0)


def stop_audit(data, entries):
    """Diagnostic counterfactuals on identical original entries, never input to signals."""
    spec = StrategySpec("cvd_rules", "fixed_1pct", 4)
    original = evaluate_strategy(data, entries, spec, risk_fraction=None)
    records = []
    for trade in original["trades"]:
        if trade["reason"] != "STOP_LOSS":
            continue
        entry, side = trade["entry"], trade["side"]
        record = dict(entry=entry, side=side, baseline_net_return=trade["net_return"])
        for mode in EXIT_MODES:
            stop, take = data.distances(np.array([entry]), mode)
            run = simulate(data, np.array([entry]), np.array([side]), replace(data.config, horizon=4),
                           stop_distances=stop, take_distances=take)
            record[mode + "_net_return"] = run["trades"][0]["net_return"]
        for horizon in (4, 24):
            entry_fill = trade["entry_fill"]
            exit_fill = data.frame.spot_close.iloc[entry + horizon - 1] * (1 - side * data.config.slippage)
            ratio = exit_fill / entry_fill
            record[f"time_only_h{horizon}_net_return"] = side * (ratio - 1) - data.config.fee * (1 + ratio)
        records.append(record)
    entry_indices = np.array([trade["entry"] for trade in original["trades"]], dtype=int)
    atr_pct = data.indicators.atr_pct.to_numpy()[entries - 1]
    active_atr = data.indicators.atr_pct.to_numpy()[entry_indices - 1] if len(entry_indices) else np.array([])
    quantiles = lambda values: np.quantile(values, [0.1, 0.5, 0.9]).tolist() if len(values) else []
    summary = dict(baseline=original["metrics"], atr_pct_quantiles=quantiles(atr_pct),
                   baseline_stop_in_atr_quantiles=quantiles(0.01 / active_atr),
                   baseline_entries_stop_below_one_atr=float((active_atr > 0.01).mean()) if len(active_atr) else None,
                   stopped_trades=len(records), counterfactuals={})
    for mode in (*EXIT_MODES, "time_only_h4", "time_only_h24"):
        key = mode + "_net_return"
        values = np.array([record[key] for record in records])
        differences = values - np.array([record["baseline_net_return"] for record in records])
        summary["counterfactuals"][mode] = dict(
            profitable=int((values > 0).sum()), mean_net_return=float(values.mean()) if len(values) else None,
            improved=int((differences > 1e-12).sum()), worsened=int((differences < -1e-12).sum()),
            mean_change=float(differences.mean()) if len(values) else None)
    return summary, records


def audit_trade_ledger(data, ledger):
    """Replay paired legacy V3 entries with original zero-slippage cost assumptions."""
    required = {"timestamp", "action", "price", "pnl", "entry_notional", "exit_reason"}
    if not required.issubset(ledger.columns) or not len(ledger) or len(ledger) % 2:
        raise ValueError("paired legacy OPEN/CLOSE trade ledger required")
    ledger = ledger.copy().reset_index(drop=True)
    ledger["timestamp"] = pd.to_datetime(ledger.timestamp, utc=True, errors="raise")
    lookup = pd.Series(np.arange(len(data.frame)), index=pd.DatetimeIndex(data.frame.timestamp))
    config = replace(data.config, horizon=4, stop_loss=0.01, take_profit=0.02, slippage=0)
    records = []
    for offset in range(0, len(ledger), 2):
        opening, closing = ledger.iloc[offset], ledger.iloc[offset + 1]
        if (opening.action not in ("OPEN_LONG", "OPEN_SHORT") or
                closing.action != opening.action.replace("OPEN_", "CLOSE_")):
            raise ValueError("ledger must contain alternating matching OPEN/CLOSE pairs")
        if opening.timestamp not in lookup or closing.timestamp not in lookup:
            raise ValueError("trade timestamp absent from market history")
        entry, exit_index = int(lookup[opening.timestamp]), int(lookup[closing.timestamp])
        side = 1 if opening.action == "OPEN_LONG" else -1
        if entry < 201 or entry + 3 >= len(data.frame) or not np.isclose(opening.price, data.frame.spot_open.iloc[entry]):
            raise ValueError("ledger entry does not match supported next-open market history")
        if (not np.isfinite(closing.entry_notional) or closing.entry_notional <= 0 or not np.isfinite(closing.pnl)):
            raise ValueError("ledger must contain finite net PnL and positive entry notional")
        recorded = float(closing.pnl / closing.entry_notional)
        indices, actions = np.array([entry]), np.array([side])
        replay = simulate(data, indices, actions, config)["trades"][0]
        if (replay["exit"] != exit_index or replay["reason"] != closing.exit_reason or
                not np.isclose(replay["exit_fill"], closing.price, atol=1e-6, rtol=1e-9) or
                not np.isclose(replay["net_return"], recorded, atol=1e-8, rtol=1e-6)):
            raise ValueError("legacy ledger cannot be reproduced under declared 1%/2%/4h/0.1%-fee assumptions")
        atr_pct = float(data.indicators.atr_pct.iloc[entry - 1])
        row = dict(entry=entry, side=side, reason=replay["reason"], atr_pct=atr_pct,
                   stop_in_atr=0.01 / atr_pct, baseline_net_return=recorded)
        for mode in EXIT_MODES:
            stop, take = data.distances(indices, mode)
            result = simulate(data, indices, actions, config, stop_distances=stop, take_distances=take)
            row[mode + "_net_return"] = result["trades"][0]["net_return"]
        for horizon in (4, 24):
            if entry + horizon <= len(data.frame):
                ratio = data.frame.spot_close.iloc[entry + horizon - 1] / opening.price
                row[f"time_only_h{horizon}_net_return"] = side * (ratio - 1) - config.fee * (1 + ratio)
            else:
                row[f"time_only_h{horizon}_net_return"] = None
        records.append(row)
    stopped = [record for record in records if record["reason"] == "STOP_LOSS"]
    summary = dict(trades=len(records), exact_replay=True, slippage=0.0,
                   exit_counts={reason: sum(record["reason"] == reason for record in records)
                                for reason in ("STOP_LOSS", "TAKE_PROFIT", "MAX_HOLD")},
                   stop_in_atr_quantiles=np.quantile([record["stop_in_atr"] for record in records], [0.1, 0.5, 0.9]).tolist(),
                   fraction_stop_below_one_atr=float(np.mean([record["stop_in_atr"] < 1 for record in records])),
                   stopped_counterfactuals={},
                   limitation="same-entry isolated trades only; counterfactual returns are not a portfolio with overlapping entries")
    for mode in (*EXIT_MODES, "time_only_h4", "time_only_h24"):
        valid = [record for record in stopped if record[mode + "_net_return"] is not None]
        values = np.array([record[mode + "_net_return"] for record in valid])
        delta = values - np.array([record["baseline_net_return"] for record in valid])
        summary["stopped_counterfactuals"][mode] = dict(
            samples=len(valid), profitable=int((values > 0).sum()), improved=int((delta > 1e-10).sum()),
            worsened=int((delta < -1e-10).sum()), mean_change=float(delta.mean()) if len(delta) else None)
    return summary, records
