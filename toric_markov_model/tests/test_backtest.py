import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import pandas as pd
import torch

from tests.test_model import tiny_model

script = Path(__file__).resolve().parents[1] / "scripts" / "backtest_trading_v3.py"
spec = importlib.util.spec_from_file_location("backtest_script", script)
backtest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest_module)
TradingBacktest = backtest_module.TradingBacktest


class Bars:
    seq_len = 1

    def __init__(self, high=103, low=97, close=100):
        self.spot_open = pd.Series([100.0, 100.0, 100.0, 100.0])
        self.spot_high = pd.Series([100.0, high, 100.0, 100.0])
        self.spot_low = pd.Series([100.0, low, 100.0, 100.0])
        self.spot_close = pd.Series([100.0, close, 100.0, 100.0])
        self.timestamps = pd.Series(pd.date_range("2025-01-01", periods=4, freq="h"))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return torch.zeros(1, 28), torch.zeros(17)


class BacktestTests(unittest.TestCase):
    def engine(self, short=False, **kwargs):
        model = tiny_model(predict_return=False).eval()
        with torch.no_grad():
            for head in (model.pattern_head, model.non_hold_gate_head):
                for parameter in head.parameters():
                    parameter.zero_()
            model.pattern_head[-1].bias.fill_(-10)
            model.pattern_head[-1].bias[int(short)] = 10
            model.non_hold_gate_head[-1].bias.fill_(10)
        return TradingBacktest(model, enable_short=short, signal_threshold=0,
                                max_hold_bars=kwargs.pop("max_hold_bars", 4), **kwargs)

    def run_quietly(self, engine, bars):
        with contextlib.redirect_stdout(io.StringIO()):
            return engine.run(bars)

    def test_entry_candle_stop_for_long_and_short(self):
        for short in (False, True):
            with self.subTest(short=short):
                engine = self.engine(short=short)
                self.run_quietly(engine, Bars())
                self.assertEqual(len(engine.trades), 2)
                opened, closed = engine.trades
                self.assertEqual(opened["timestamp"], closed["timestamp"])
                self.assertEqual(closed["exit_reason"], "STOP_LOSS")
                self.assertLess(closed["pnl"], 0)

    def test_entry_candle_take_profit(self):
        engine = self.engine()
        self.run_quietly(engine, Bars(low=99.5))
        self.assertEqual(engine.trades[1]["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(engine.trades[0]["timestamp"], engine.trades[1]["timestamp"])

    def test_one_bar_horizon_exits_on_entry_close(self):
        engine = self.engine(max_hold_bars=1)
        self.run_quietly(engine, Bars(high=100.5, low=99.5, close=100.2))
        self.assertEqual(engine.trades[1]["exit_reason"], "MAX_HOLD")
        self.assertEqual(engine.trades[1]["price"], 100.2)

    def test_holding_uses_tail_bars_without_new_entries(self):
        engine = self.engine(max_hold_bars=3)
        bars = Bars(high=100.5, low=99.5)
        self.run_quietly(engine, bars)
        self.assertEqual(engine.trades[1]["timestamp"], bars.timestamps.iloc[-1])
        self.assertEqual(engine.trades[1]["exit_reason"], "MAX_HOLD")

    def test_gap_stop_fills_at_open(self):
        engine = self.engine()
        engine.position_side, engine.entry_price = 1, 100
        should_exit, price, reason = engine.check_exit_conditions(97, 90, 94, 95)
        self.assertTrue(should_exit)
        self.assertEqual((price, reason), (95, "STOP_LOSS"))
