#!/usr/bin/env python3
"""Backtesting script for trading model V3 with Basis and Open Interest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda/cpu")
    parser.add_argument("--train-split", type=float, default=None, help="Override train/val split used inside dataset")
    parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--position-size", type=float, default=0.95, help="Fraction of capital per trade")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost (0.1%)")
    parser.add_argument("--confidence-threshold", type=float, default=0.50, help="Min confidence to trade (0.0-1.0)")
    parser.add_argument("--pattern-prob-threshold", type=float, default=0.30, help="Min pattern probability to treat signal as valid")
    parser.add_argument("--signal-threshold", type=float, default=0.22, help="Min (probability * confidence) score for entry")
    parser.add_argument("--use-checkpoint-pattern-threshold", action="store_true", help="Use best_val_binary_non_hold_threshold from checkpoint as pattern-prob-threshold")
    parser.add_argument("--cooldown-bars", type=int, default=1, help="Bars to wait after closing position")
    parser.add_argument("--max-hold-bars", type=int, default=96, help="Force-close position after N bars (0 to disable)")
    parser.add_argument("--take-profit", type=float, default=0.02, help="Take profit percentage (default 2%)")
    parser.add_argument("--stop-loss", type=float, default=0.01, help="Stop loss percentage (default 1%)")
    parser.add_argument("--intrabar-priority", type=str, default="stop_first", choices=["stop_first", "take_first"], help="If both TP and SL hit in same bar")
    parser.add_argument("--enable-short", action="store_true", help="Enable short entries on bearish patterns")
    parser.add_argument("--dynamic-position-sizing", action="store_true", help="Use ret/vol heads to adapt position size")
    parser.add_argument("--min-position-size", type=float, default=0.20, help="Min position fraction when dynamic sizing is on")
    parser.add_argument("--size-edge-scale", type=float, default=3.0, help="Edge scaling for dynamic sizing")
    parser.add_argument("--size-vol-scale", type=float, default=1.5, help="Volume penalty scale for dynamic sizing")
    parser.add_argument("--bars-per-year", type=float, default=24 * 365, help="Bars/year for Sharpe annualization")
    parser.add_argument("--optimize-signal-threshold", action="store_true", help="Grid-search signal threshold by backtest metric")
    parser.add_argument("--threshold-min", type=float, default=0.10, help="Min threshold for optimization")
    parser.add_argument("--threshold-max", type=float, default=0.90, help="Max threshold for optimization")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Threshold step for optimization")
    parser.add_argument("--optimization-metric", type=str, default="profit_factor", choices=["profit_factor", "total_return", "mar"], help="Metric used for threshold optimization")
    parser.add_argument("--min-round-trips-for-optimization", type=int, default=5, help="Ignore thresholds with fewer completed trades")
    parser.add_argument("--output", type=str, default="backtest_results_v3.csv", help="Output CSV path")
    return parser.parse_args()


class TradingBacktest:
    """Backtesting engine for trading model."""

    ACTION_OPEN_LONG = 2
    ACTION_OPEN_SHORT = -2
    ACTION_CLOSE = 0
    ACTION_HOLD = 1

    PATTERN_NAMES = [
        "Bullish Div", "Bearish Div", "CVD Rev Bull", "CVD Rev Bear",
        "CVD Exh Bull", "CVD Exh Bear", "CVD SF Bull", "CVD SF Bear",
        "CVD Spike Bull", "CVD Spike Bear", "Basis Long", "Basis Short",
        "Accumulation", "Distribution", "POC Break Up", "POC Break Down", "Hold",
    ]
    BULLISH_PATTERNS = {0, 2, 4, 6, 8, 10, 12, 14}
    BEARISH_PATTERNS = {1, 3, 5, 7, 9, 11, 13, 15}

    def __init__(
        self,
        model: ToricTradingModelV3,
        initial_capital: float = 10000,
        position_size: float = 0.95,
        transaction_cost: float = 0.001,
        confidence_threshold: float = 0.50,
        pattern_prob_threshold: float = 0.30,
        signal_threshold: float = 0.22,
        cooldown_bars: int = 1,
        max_hold_bars: int = 96,
        take_profit_pct: float = 0.02,
        stop_loss_pct: float = 0.01,
        intrabar_priority: str = "stop_first",
        enable_short: bool = False,
        dynamic_position_sizing: bool = False,
        min_position_size: float = 0.20,
        size_edge_scale: float = 3.0,
        size_vol_scale: float = 1.5,
        bars_per_year: float = 24 * 365,
    ):
        self.model = model
        self.initial_capital = float(initial_capital)
        self.position_size = float(position_size)
        self.transaction_cost = float(transaction_cost)
        self.confidence_threshold = float(confidence_threshold)
        self.pattern_prob_threshold = float(pattern_prob_threshold)
        self.signal_threshold = float(signal_threshold)
        self.cooldown_bars = int(cooldown_bars)
        self.max_hold_bars = int(max_hold_bars)
        self.take_profit_pct = float(take_profit_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.intrabar_priority = intrabar_priority
        self.enable_short = bool(enable_short)
        self.dynamic_position_sizing = bool(dynamic_position_sizing)
        self.min_position_size = float(min_position_size)
        self.size_edge_scale = float(size_edge_scale)
        self.size_vol_scale = float(size_vol_scale)
        self.bars_per_year = float(bars_per_year)

        self.reset()

    def reset(self) -> None:
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.position_side = 0
        self.position_units = 0.0
        self.position_notional = 0.0
        self.position_entry_fee = 0.0
        self.entry_price = 0.0
        self.entry_pattern = ""
        self.position_bars = 0
        self.cooldown_remaining = 0
        self.trades: list[dict[str, Any]] = []
        self.portfolio_values: list[dict[str, Any]] = []

    def _current_unrealized_pnl(self, mark_price: float) -> float:
        if self.position_side == 0:
            return 0.0
        return float(self.position_side * self.position_units * (mark_price - self.entry_price))

    def _current_equity(self, mark_price: float) -> float:
        return float(self.capital + self._current_unrealized_pnl(mark_price))

    def _record_portfolio(self, timestamp: Any, mark_price: float) -> None:
        unrealized = self._current_unrealized_pnl(mark_price)
        total_value = self.capital + unrealized
        self.portfolio_values.append(
            {
                "timestamp": timestamp,
                "capital": float(self.capital),
                "position_side": int(self.position_side),
                "position_units": float(self.position_units),
                "entry_price": float(self.entry_price),
                "position_notional": float(self.position_notional),
                "unrealized_pnl": float(unrealized),
                "total_value": float(total_value),
                "price": float(mark_price),
            }
        )

    def _compute_position_size(self, result: dict[str, torch.Tensor]) -> float:
        base = float(np.clip(self.position_size, 0.0, 1.0))
        if not self.dynamic_position_sizing:
            return base
        if "predicted_return" not in result or "predicted_volume_change" not in result:
            return base

        pred_ret = abs(float(result["predicted_return"].item()))
        pred_vol = abs(float(result["predicted_volume_change"].item()))

        edge_term = np.tanh(self.size_edge_scale * pred_ret)
        vol_penalty = 1.0 + self.size_vol_scale * pred_vol
        scaled = base * edge_term / vol_penalty

        min_size = float(np.clip(self.min_position_size, 0.0, base))
        return float(np.clip(scaled, min_size, base))

    def _open_position(
        self,
        side: int,
        price: float,
        timestamp: Any,
        pattern_name: str,
        best_prob: float,
        best_confidence: float,
        best_score: float,
        size_fraction: float,
    ) -> bool:
        if self.position_side != 0 or side not in (-1, 1):
            return False

        size_fraction = float(np.clip(size_fraction, 0.0, 1.0))
        notional = self.capital * size_fraction
        if notional <= 0.0:
            return False

        entry_fee = notional * self.transaction_cost
        if entry_fee >= self.capital:
            return False

        self.capital -= entry_fee
        self.position_side = side
        self.position_notional = notional
        self.position_entry_fee = entry_fee
        self.position_units = notional / price
        self.entry_price = price
        self.entry_pattern = pattern_name
        self.position_bars = 0

        self.trades.append(
            {
                "timestamp": timestamp,
                "action": "OPEN_LONG" if side > 0 else "OPEN_SHORT",
                "pattern": pattern_name,
                "price": float(price),
                "units": float(self.position_units),
                "notional": float(notional),
                "entry_fee": float(entry_fee),
                "best_prob": float(best_prob),
                "best_confidence": float(best_confidence),
                "best_score": float(best_score),
                "size_fraction": float(size_fraction),
                "capital": float(self.capital),
            }
        )
        return True

    def _close_position(self, price: float, timestamp: Any, exit_reason: str) -> bool:
        if self.position_side == 0:
            return False

        gross_pnl = self.position_side * self.position_units * (price - self.entry_price)
        exit_notional = self.position_units * price
        exit_fee = exit_notional * self.transaction_cost

        # Capital already includes entry fee deduction from open.
        self.capital += gross_pnl - exit_fee

        fees_total = self.position_entry_fee + exit_fee
        trade_pnl = gross_pnl - fees_total
        side_label = "LONG" if self.position_side > 0 else "SHORT"
        directional_move_pct = self.position_side * (price - self.entry_price) / self.entry_price * 100.0
        notional_pct = trade_pnl / (self.position_notional + 1e-8) * 100.0

        self.trades.append(
            {
                "timestamp": timestamp,
                "action": f"CLOSE_{side_label}",
                "pattern": self.entry_pattern,
                "exit_reason": exit_reason,
                "price": float(price),
                "entry_price": float(self.entry_price),
                "units": float(self.position_units),
                "entry_notional": float(self.position_notional),
                "gross_pnl": float(gross_pnl),
                "pnl": float(trade_pnl),
                "pnl_pct": float(notional_pct),
                "directional_move_pct": float(directional_move_pct),
                "entry_fee": float(self.position_entry_fee),
                "exit_fee": float(exit_fee),
                "fees_total": float(fees_total),
                "capital": float(self.capital),
            }
        )

        self.position_side = 0
        self.position_units = 0.0
        self.position_notional = 0.0
        self.position_entry_fee = 0.0
        self.entry_price = 0.0
        self.entry_pattern = ""
        self.position_bars = 0
        self.cooldown_remaining = self.cooldown_bars
        return True

    def check_exit_conditions(self, bar_high: float, bar_low: float, bar_close: float) -> tuple[bool, float, str]:
        """Check TP/SL and max-hold using intrabar high/low range."""
        if self.position_side == 0:
            return False, bar_close, ""

        if self.position_side > 0:
            tp_price = self.entry_price * (1.0 + self.take_profit_pct)
            sl_price = self.entry_price * (1.0 - self.stop_loss_pct)
            hit_tp = bar_high >= tp_price
            hit_sl = bar_low <= sl_price
        else:
            tp_price = self.entry_price * (1.0 - self.take_profit_pct)
            sl_price = self.entry_price * (1.0 + self.stop_loss_pct)
            hit_tp = bar_low <= tp_price
            hit_sl = bar_high >= sl_price

        if hit_tp and hit_sl:
            if self.intrabar_priority == "stop_first":
                return True, float(sl_price), "STOP_LOSS"
            return True, float(tp_price), "TAKE_PROFIT"
        if hit_tp:
            return True, float(tp_price), "TAKE_PROFIT"
        if hit_sl:
            return True, float(sl_price), "STOP_LOSS"

        if self.max_hold_bars > 0 and self.position_bars >= self.max_hold_bars:
            return True, float(bar_close), "MAX_HOLD"

        return False, float(bar_close), ""

    def run(self, dataset: TradingDatasetV3) -> dict[str, float]:
        """Run backtest on dataset with PATTERN DETECTION."""
        self.reset()
        device = next(self.model.parameters()).device

        print(f"Running backtest on {len(dataset)} samples...")
        print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        print(f"Pattern probability threshold: {self.pattern_prob_threshold:.2f}")
        print(f"Signal score threshold: {self.signal_threshold:.2f}")
        print(f"Cooldown bars after exit: {self.cooldown_bars}")
        print(f"Short enabled: {self.enable_short}")
        print(f"Dynamic position sizing: {self.dynamic_position_sizing}")
        if self.max_hold_bars > 0:
            print(f"Max hold bars: {self.max_hold_bars}")

        counters = {
            "signals_detected": 0,
            "opened": 0,
            "skipped_no_pattern": 0,
            "skipped_cooldown": 0,
            "skipped_bearish": 0,
            "skipped_low_score": 0,
        }

        last_timestamp: Any | None = None
        last_close_price: float | None = None
        market_len = len(dataset.spot_open)

        for i in range(len(dataset)):
            bar_idx = i + dataset.seq_len
            if bar_idx >= market_len:
                break

            open_price = float(dataset.spot_open.iloc[bar_idx])
            high_price = float(dataset.spot_high.iloc[bar_idx])
            low_price = float(dataset.spot_low.iloc[bar_idx])
            close_price = float(dataset.spot_close.iloc[bar_idx])
            timestamp = dataset.timestamps.iloc[bar_idx]
            last_timestamp = timestamp
            last_close_price = close_price

            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1

            if self.position_side != 0:
                self.position_bars += 1
                should_exit, exit_price, exit_reason = self.check_exit_conditions(high_price, low_price, close_price)
                if should_exit:
                    self._close_position(exit_price, timestamp, exit_reason)
                self._record_portfolio(timestamp, close_price)

                if (i + 1) % 100 == 0:
                    current_value = self.portfolio_values[-1]["total_value"]
                    side_name = "LONG" if self.position_side > 0 else ("SHORT" if self.position_side < 0 else "FLAT")
                    print(
                        f"  Step {i + 1}/{len(dataset)}: "
                        f"Portfolio=${current_value:.2f}, "
                        f"Return={100 * (current_value / self.initial_capital - 1):.2f}%, "
                        f"Trades={len(self.trades)}, "
                        f"Position={side_name}"
                    )
                continue

            features, _ = dataset[i]
            features = features.unsqueeze(0).to(device)

            with torch.no_grad():
                result = self.model.detect_patterns(
                    features,
                    confidence_threshold=self.confidence_threshold,
                    pattern_prob_threshold=self.pattern_prob_threshold,
                )

            best_pattern = int(result["best_non_hold_pattern"].item())
            best_prob = float(result["best_non_hold_prob"].item())
            best_confidence = float(result["best_non_hold_confidence"].item())
            best_score = float(result["best_non_hold_score"].item())
            hold_score = float(result["hold_score"].item())
            has_pattern = bool(result["has_pattern"].item())

            action = self.ACTION_HOLD
            action_side = 0
            pattern_name = "No pattern"

            if self.cooldown_remaining > 0:
                counters["skipped_cooldown"] += 1
            elif not has_pattern:
                counters["skipped_no_pattern"] += 1
            else:
                counters["signals_detected"] += 1
                pattern_name = self.PATTERN_NAMES[best_pattern]
                if best_score < self.signal_threshold:
                    counters["skipped_low_score"] += 1
                elif best_pattern in self.BULLISH_PATTERNS:
                    action = self.ACTION_OPEN_LONG
                    action_side = 1
                elif best_pattern in self.BEARISH_PATTERNS:
                    if self.enable_short:
                        action = self.ACTION_OPEN_SHORT
                        action_side = -1
                    else:
                        counters["skipped_bearish"] += 1

            if action in (self.ACTION_OPEN_LONG, self.ACTION_OPEN_SHORT):
                size_fraction = self._compute_position_size(result)
                opened = self._open_position(
                    side=action_side,
                    price=open_price,
                    timestamp=timestamp,
                    pattern_name=pattern_name,
                    best_prob=best_prob,
                    best_confidence=best_confidence,
                    best_score=best_score,
                    size_fraction=size_fraction,
                )
                if opened:
                    counters["opened"] += 1

            self._record_portfolio(timestamp, close_price)

            if (i + 1) % 100 == 0:
                current_value = self.portfolio_values[-1]["total_value"]
                side_name = "LONG" if self.position_side > 0 else ("SHORT" if self.position_side < 0 else "FLAT")
                print(
                    f"  Step {i + 1}/{len(dataset)}: "
                    f"Portfolio=${current_value:.2f}, "
                    f"Return={100 * (current_value / self.initial_capital - 1):.2f}%, "
                    f"Trades={len(self.trades)}, "
                    f"Position={side_name}"
                )
                if has_pattern:
                    print(
                        f"    Last pattern: {pattern_name} "
                        f"(prob={best_prob:.2f}, conf={best_confidence:.2f}, score={best_score:.2f}, hold_score={hold_score:.2f})"
                    )

        if self.position_side != 0 and last_timestamp is not None and last_close_price is not None:
            self._close_position(last_close_price, last_timestamp, "END_OF_TEST")
            self._record_portfolio(last_timestamp, last_close_price)

        print(f"\nSignals passing base gates: {counters['signals_detected']}")
        print(f"Opened positions: {counters['opened']}")
        print(f"Skipped by cooldown: {counters['skipped_cooldown']}")
        print(f"Skipped bearish patterns (short disabled): {counters['skipped_bearish']}")
        print(f"Skipped low-score signals: {counters['skipped_low_score']}")
        print(f"Skipped with no pattern: {counters['skipped_no_pattern']}")
        print(f"Total trade events: {len(self.trades)}")

        return self.get_metrics()

    def get_metrics(self) -> dict[str, float]:
        """Calculate performance metrics, including profit factor."""
        if not self.portfolio_values:
            return {}

        values = np.asarray([pv["total_value"] for pv in self.portfolio_values], dtype=np.float64)
        returns = np.diff(values) / np.clip(values[:-1], 1e-8, None)

        final_value = float(values[-1])
        total_return = float((final_value / self.initial_capital - 1.0) * 100.0)

        if len(returns) > 1 and float(returns.std()) > 0.0:
            sharpe_ratio = float(np.sqrt(self.bars_per_year) * returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0

        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / np.clip(peak, 1e-8, None)
        max_drawdown = float(drawdown.min() * 100.0)

        close_trades = [t for t in self.trades if str(t.get("action", "")).startswith("CLOSE_")]
        open_trades = [t for t in self.trades if str(t.get("action", "")).startswith("OPEN_")]
        trade_pnls = np.asarray([float(t.get("pnl", 0.0)) for t in close_trades], dtype=np.float64)

        gross_profit = float(trade_pnls[trade_pnls > 0].sum()) if len(trade_pnls) else 0.0
        gross_loss = float(np.abs(trade_pnls[trade_pnls < 0].sum())) if len(trade_pnls) else 0.0
        if gross_loss > 0.0:
            profit_factor = float(gross_profit / gross_loss)
        else:
            profit_factor = float(np.inf if gross_profit > 0.0 else 0.0)

        win_mask = trade_pnls > 0
        loss_mask = trade_pnls < 0
        win_rate = float(100.0 * win_mask.mean()) if len(trade_pnls) else 0.0
        avg_trade = float(trade_pnls.mean()) if len(trade_pnls) else 0.0
        avg_win = float(trade_pnls[win_mask].mean()) if np.any(win_mask) else 0.0
        avg_loss = float(trade_pnls[loss_mask].mean()) if np.any(loss_mask) else 0.0
        payoff_ratio = float(abs(avg_win / avg_loss)) if avg_loss < 0 else 0.0

        total_fees = float(sum(float(t.get("fees_total", 0.0)) for t in close_trades))
        mar_ratio = float(total_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0

        exit_reasons: dict[str, int] = {}
        for t in close_trades:
            reason = str(t.get("exit_reason", "UNKNOWN"))
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        metrics = {
            "initial_capital": float(self.initial_capital),
            "final_value": final_value,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "mar_ratio": mar_ratio,
            "num_trade_events": float(len(self.trades)),
            "num_entries": float(len(open_trades)),
            "round_trip_trades": float(len(close_trades)),
            "long_entries": float(sum(1 for t in open_trades if t.get("action") == "OPEN_LONG")),
            "short_entries": float(sum(1 for t in open_trades if t.get("action") == "OPEN_SHORT")),
            "win_rate_pct": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_trade_pnl": avg_trade,
            "avg_win_pnl": avg_win,
            "avg_loss_pnl": avg_loss,
            "payoff_ratio": payoff_ratio,
            "total_fees": total_fees,
            "num_wins": float(win_mask.sum()) if len(trade_pnls) else 0.0,
            "num_losses": float(loss_mask.sum()) if len(trade_pnls) else 0.0,
            "num_breakeven": float((trade_pnls == 0).sum()) if len(trade_pnls) else 0.0,
            # Backward compatibility keys
            "num_trades": float(len(self.trades)),
        }

        for reason, count in exit_reasons.items():
            metrics[f"exit_{reason.lower()}"] = float(count)

        return metrics


def threshold_grid_search(
    dataset: TradingDatasetV3,
    backtest_kwargs: dict[str, Any],
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    metric_name: str,
    min_round_trips: int,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    thresholds = np.arange(threshold_min, threshold_max + 1e-9, threshold_step)
    rows: list[dict[str, float]] = []

    best_threshold = float(threshold_min)
    best_metrics: dict[str, float] = {}
    best_score = -np.inf

    for threshold in thresholds:
        bt = TradingBacktest(**backtest_kwargs, signal_threshold=float(threshold))
        metrics = bt.run(dataset)

        profit_factor = float(metrics.get("profit_factor", 0.0))
        total_return = float(metrics.get("total_return_pct", 0.0))
        mar_ratio = float(metrics.get("mar_ratio", 0.0))
        round_trips = int(metrics.get("round_trip_trades", 0.0))

        if round_trips < min_round_trips:
            score = -np.inf
        elif metric_name == "profit_factor":
            score = 1e9 if np.isinf(profit_factor) else profit_factor
        elif metric_name == "total_return":
            score = total_return
        elif metric_name == "mar":
            score = mar_ratio
        else:
            score = total_return

        rows.append(
            {
                "signal_threshold": float(threshold),
                "score": float(score if np.isfinite(score) else -1.0),
                "profit_factor": profit_factor,
                "total_return_pct": total_return,
                "mar_ratio": mar_ratio,
                "round_trip_trades": float(round_trips),
                "win_rate_pct": float(metrics.get("win_rate_pct", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
            }
        )

        is_better = score > best_score
        if is_better:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    scan_df = pd.DataFrame(rows)
    return best_threshold, best_metrics, scan_df


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    print("=" * 80)
    print("BACKTEST V3 MODEL WITH PATTERN DETECTION")
    print("=" * 80)
    print("Loading checkpoint...")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]

    if args.use_checkpoint_pattern_threshold:
        best_val_threshold = checkpoint.get("best_val_binary_non_hold_threshold")
        if best_val_threshold is not None:
            args.pattern_prob_threshold = float(best_val_threshold)
            print(
                "Using checkpoint pattern threshold: "
                f"pattern_prob_threshold={args.pattern_prob_threshold:.2f}"
            )

    normalization = checkpoint.get("normalization")
    norm_stats = None
    if normalization is not None:
        norm_stats = {
            "feature_mean": normalization["feature_mean"].detach().cpu().numpy(),
            "feature_std": normalization["feature_std"].detach().cpu().numpy(),
        }

    train_split = args.train_split
    if train_split is None:
        train_split = float(getattr(ckpt_args, "train_split", 0.8))

    print("Creating V3 PATTERN DETECTION model...")
    try:
        test_dataset = TradingDatasetV3(
            csv_path=args.data,
            seq_len=ckpt_args.seq_len,
            prediction_horizon=ckpt_args.prediction_horizon,
            train=False,
            train_split=train_split,
            min_pattern_profit=getattr(ckpt_args, "min_pattern_profit", 0.003),
            normalization_stats=norm_stats,
        )
    except ValueError as exc:
        if "Normalization stats feature dimension mismatch" in str(exc):
            raw_dataset = TradingDatasetV3(
                csv_path=args.data,
                seq_len=ckpt_args.seq_len,
                prediction_horizon=ckpt_args.prediction_horizon,
                train=False,
                train_split=train_split,
                min_pattern_profit=getattr(ckpt_args, "min_pattern_profit", 0.003),
                normalization_stats=None,
            )
            ckpt_dim = int(len(norm_stats["feature_mean"])) if norm_stats is not None else -1
            data_dim = int(raw_dataset.features.shape[1])
            raise RuntimeError(
                "Checkpoint/data feature mismatch. "
                f"Checkpoint expects {ckpt_dim} features, current dataset builds {data_dim}. "
                "Use a checkpoint retrained with current feature set."
            ) from exc
        raise
    num_features = test_dataset.features.shape[1]
    print(f"Model expects {num_features} features")

    model = ToricTradingModelV3(
        num_features=num_features,
        dim_angles=ckpt_args.dim_angles,
        max_len=ckpt_args.seq_len,
        num_states=ckpt_args.num_states,
        num_levels=4,
        num_layers=ckpt_args.num_layers,
        n_bits=8,
        use_attention=True,
        num_patterns=17,
        predict_return=True,
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint into current model architecture. "
            "Most likely feature dimension or layer config mismatch."
        ) from exc
    model.eval()

    print(
        f"Loaded model from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.4f}, "
        f"val_acc={checkpoint['val_accuracy']:.2f}%)"
    )
    if "val_non_hold_f1" in checkpoint:
        print(
            f"Checkpoint non-hold F1={checkpoint['val_non_hold_f1']:.2f}% "
            f"(P={checkpoint.get('val_non_hold_precision', 0.0):.2f}%, "
            f"R={checkpoint.get('val_non_hold_recall', 0.0):.2f}%)"
        )

    print("\nRunning backtest...")
    print("=" * 80)

    backtest_kwargs: dict[str, Any] = {
        "model": model,
        "initial_capital": args.initial_capital,
        "position_size": args.position_size,
        "transaction_cost": args.transaction_cost,
        "confidence_threshold": args.confidence_threshold,
        "pattern_prob_threshold": args.pattern_prob_threshold,
        "cooldown_bars": args.cooldown_bars,
        "max_hold_bars": args.max_hold_bars,
        "take_profit_pct": args.take_profit,
        "stop_loss_pct": args.stop_loss,
        "intrabar_priority": args.intrabar_priority,
        "enable_short": args.enable_short,
        "dynamic_position_sizing": args.dynamic_position_sizing,
        "min_position_size": args.min_position_size,
        "size_edge_scale": args.size_edge_scale,
        "size_vol_scale": args.size_vol_scale,
        "bars_per_year": args.bars_per_year,
    }

    metrics: dict[str, float]
    chosen_signal_threshold = args.signal_threshold

    if args.optimize_signal_threshold:
        print(
            f"Grid search signal threshold: [{args.threshold_min:.2f}, {args.threshold_max:.2f}] "
            f"step={args.threshold_step:.2f}, metric={args.optimization_metric}"
        )
        best_threshold, best_metrics, scan_df = threshold_grid_search(
            dataset=test_dataset,
            backtest_kwargs=backtest_kwargs,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            metric_name=args.optimization_metric,
            min_round_trips=args.min_round_trips_for_optimization,
        )
        chosen_signal_threshold = best_threshold

        scan_output = Path(args.output).with_suffix("")
        scan_csv = f"{scan_output}_threshold_scan.csv"
        scan_df.to_csv(scan_csv, index=False)
        print(f"Threshold scan saved to: {scan_csv}")
        print(f"Best signal threshold by {args.optimization_metric}: {best_threshold:.2f}")

        # Re-run once to capture trades/portfolio exactly for selected threshold output files.
        backtest = TradingBacktest(**backtest_kwargs, signal_threshold=chosen_signal_threshold)
        metrics = backtest.run(test_dataset)
    else:
        backtest = TradingBacktest(**backtest_kwargs, signal_threshold=chosen_signal_threshold)
        metrics = backtest.run(test_dataset)

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Signal Threshold:   {chosen_signal_threshold:.2f}")
    print(f"Initial Capital:   ${metrics.get('initial_capital', 0.0):,.2f}")
    print(f"Final Value:       ${metrics.get('final_value', 0.0):,.2f}")
    print(f"Total Return:      {metrics.get('total_return_pct', 0.0):+.2f}%")
    print(f"Profit Factor:     {metrics.get('profit_factor', 0.0):.4f}")
    print(f"Gross Profit:      ${metrics.get('gross_profit', 0.0):,.2f}")
    print(f"Gross Loss:        ${metrics.get('gross_loss', 0.0):,.2f}")
    print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"Max Drawdown:      {metrics.get('max_drawdown_pct', 0.0):.2f}%")
    print(f"Round Trips:       {int(metrics.get('round_trip_trades', 0.0))}")
    print(f"Trade Events:      {int(metrics.get('num_trade_events', 0.0))}")
    print(f"Win Rate:          {metrics.get('win_rate_pct', 0.0):.2f}%")
    print(f"Avg Trade PnL:     ${metrics.get('avg_trade_pnl', 0.0):,.2f}")
    print(f"Avg Win / Loss:    ${metrics.get('avg_win_pnl', 0.0):,.2f} / ${metrics.get('avg_loss_pnl', 0.0):,.2f}")
    print(f"Long / Short:      {int(metrics.get('long_entries', 0.0))} / {int(metrics.get('short_entries', 0.0))}")
    print("=" * 80)

    portfolio_df = pd.DataFrame(backtest.portfolio_values)
    portfolio_df.to_csv(args.output, index=False)
    print(f"\nPortfolio history saved to: {args.output}")

    trades_df = pd.DataFrame(backtest.trades)
    if len(trades_df) > 0:
        trades_output = args.output.replace(".csv", "_trades.csv")
        trades_df.to_csv(trades_output, index=False)
        print(f"Trade history saved to: {trades_output}")

    metrics_output = args.output.replace(".csv", "_metrics.json")
    with open(metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {metrics_output}")

    start_idx = test_dataset.seq_len
    end_idx = min(start_idx + len(test_dataset) - 1, len(test_dataset.spot_close) - 1)
    if start_idx < len(test_dataset.spot_open) and end_idx >= start_idx:
        first_price = float(test_dataset.spot_open.iloc[start_idx])
        last_price = float(test_dataset.spot_close.iloc[end_idx])
        bh_return = 100.0 * (last_price / first_price - 1.0)
        print(f"\nBuy-and-Hold Return: {bh_return:+.2f}%")
        print(f"Model vs B&H:        {metrics.get('total_return_pct', 0.0) - bh_return:+.2f}%")


if __name__ == "__main__":
    main()
