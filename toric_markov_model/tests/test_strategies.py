from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.support import market_frame
from toric_markov_model.strategies import (
    StrategyData, StrategySpec, audit_trade_ledger, calibration_passes, choose_on_selection, evaluate_strategy,
    stop_audit, strategy_grid, strategy_signals, wilder_atr,
)
from toric_markov_model.train.payoff import simulate


class StrategyTests(unittest.TestCase):
    def test_wilder_atr_includes_gap_and_uses_sma_seed(self):
        frame = pd.DataFrame(dict(spot_high=[101, 103, 111, 110], spot_low=[99, 99, 108, 106],
                                  spot_close=[100, 101, 109, 107]))
        atr = wilder_atr(frame, period=3)
        self.assertTrue(np.isnan(atr.iloc[:2]).all())
        self.assertAlmostEqual(atr.iloc[2], (2 + 4 + 10) / 3)
        self.assertAlmostEqual(atr.iloc[3], ((16 / 3) * 2 + 4) / 3)

    def test_entry_and_future_changes_cannot_change_signal_or_atr_stop(self):
        frame = market_frame(700)
        original = StrategyData.from_frame(frame)
        frame.loc[400:, "spot_cvd"] += 100000
        for field in ("open", "high", "low", "close"):
            frame.loc[400:, f"spot_{field}"] *= 2
        changed = StrategyData.from_frame(frame)
        for family in original.signals:
            np.testing.assert_array_equal(original.signals[family][:400], changed.signals[family][:400])
        np.testing.assert_array_equal(original.distances(np.array([400]), "atr_2"),
                                      changed.distances(np.array([400]), "atr_2"))

    def test_new_signal_families_have_explicit_long_and_short_triggers(self):
        frame = market_frame(300)
        indicators = StrategyData.from_frame(frame).indicators.copy()
        for name in ("ema_20", "ema_50", "ema_200", "channel_high", "channel_low", "atr", "relative_volume", "range_zscore"):
            indicators[name] = 100.0
        frame["spot_close"] = 100.0
        indicators["relative_volume"] = 2.0
        indicators["atr"] = 1.0
        indicators["range_zscore"] = 0.0
        indicators.loc[220, ["ema_50", "ema_200", "channel_high"]] = [101, 99, 102]
        frame.loc[220, "spot_close"] = 103
        indicators.loc[240, ["ema_50", "ema_200", "channel_low"]] = [99, 101, 98]
        frame.loc[240, "spot_close"] = 97
        indicators.loc[259, "range_zscore"] = -2.5
        indicators.loc[279, "range_zscore"] = 2.5
        signals = strategy_signals(frame, indicators)
        self.assertEqual(signals["trend_breakout"][220], 1)
        self.assertEqual(signals["trend_breakout"][240], -1)
        self.assertEqual(signals["trend_pullback"][220], 1)
        self.assertEqual(signals["trend_pullback"][240], -1)
        self.assertEqual(signals["range_reversion"][260], 1)
        self.assertEqual(signals["range_reversion"][280], -1)

    def test_atr_distances_are_frozen_at_entry(self):
        data = StrategyData.from_frame(market_frame(300))
        entries = np.arange(220, 240)
        actions = np.zeros(len(entries), dtype=int)
        actions[0] = 1
        stops = np.full(len(entries), 0.1)
        stops[1:] = 0.00001
        run = simulate(data, entries, actions, replace(data.config, horizon=4), stop_distances=stops,
                       take_distances=np.full(len(entries), 0.5))
        self.assertEqual(run["trades"][0]["stop_distance"], 0.1)
        repeated = simulate(data, entries, actions, stop_distances=np.full(len(entries), 0.1),
                            take_distances=np.full(len(entries), 0.5))
        self.assertEqual(run["trades"], repeated["trades"])

    def test_wider_stop_reduces_size_and_same_planned_risk(self):
        frame = market_frame(300)
        for field in ("open", "high", "low", "close"):
            frame[f"spot_{field}"] = 100.0
        data = StrategyData.from_frame(frame)
        for side in (1, -1):
            data.frame.loc[220, "spot_low"] = 95
            data.frame.loc[220, "spot_high"] = 105
            results = []
            for distance in (0.01, 0.03):
                result = simulate(data, np.array([220]), np.array([side]), stop_distances=[distance],
                                  take_distances=[0.5], risk_fraction=0.001)
                self.assertAlmostEqual(result["metrics"]["total_return_pct"], -0.1, places=10)
                results.append(result["trades"][0])
            self.assertLess(results[1]["size_fraction"], results[0]["size_fraction"])

    def test_gap_can_exceed_risk_budget_and_does_not_fill_at_stop(self):
        data = StrategyData.from_frame(market_frame(300))
        data.frame.loc[220, ["spot_open", "spot_high", "spot_low", "spot_close"]] = [100, 100, 100, 100]
        data.frame.loc[221, ["spot_open", "spot_high", "spot_low", "spot_close"]] = [90, 91, 89, 90]
        result = simulate(data, np.array([220]), np.array([1]), risk_fraction=0.001)
        self.assertLess(result["metrics"]["total_return_pct"], -0.1)
        self.assertAlmostEqual(result["trades"][0]["exit_fill"], 90 * (1 - data.config.slippage))

    def test_fixed_distances_preserve_existing_simulation(self):
        data = StrategyData.from_frame(market_frame(300))
        entries = np.arange(220, 250)
        actions = data.signals["cvd_rules"][entries - 1]
        default = simulate(data, entries, actions)
        explicit = simulate(data, entries, actions, stop_distances=np.full(len(entries), 0.01),
                            take_distances=np.full(len(entries), 0.02))
        self.assertEqual(default["metrics"], explicit["metrics"])
        self.assertEqual(default["trades"], explicit["trades"])

    def test_all_horizons_share_entry_range_and_equity_dates(self):
        data = StrategyData.from_frame(market_frame(500))
        entries = data.entries(250, 400)
        curves = [evaluate_strategy(data, entries, StrategySpec("cvd_rules", "fixed_1pct", horizon))
                  for horizon in (4, 12, 24)]
        self.assertEqual(len({len(run["equity"]) for run in curves}), 1)
        self.assertEqual(len({len(run["daily_returns"]) for run in curves}), 1)
        self.assertEqual(entries[-1] + 23, 399)

    def test_selection_and_confirmation_are_separate_and_reject_empty_policy(self):
        positive = dict(trades=20, block_lower_daily_mean=0.001, total_return_pct=1,
                        positive_calendar_halves=True)
        negative = {**positive, "block_lower_daily_mean": -0.001}
        self.assertEqual(choose_on_selection({"candidate": positive, "other": negative}), "candidate")
        self.assertIsNone(choose_on_selection({"empty": {**positive, "trades": 0}}))
        self.assertTrue(calibration_passes(positive, positive))
        self.assertFalse(calibration_passes(negative, positive))
        self.assertFalse(calibration_passes(positive, {**positive, "total_return_pct": -1}))

    def test_invalid_distances_and_grid_are_rejected(self):
        self.assertEqual(len(strategy_grid()), 48)
        data = StrategyData.from_frame(market_frame(300))
        for distances in ([float("nan")], [0], [1], [0.1, 0.2]):
            with self.assertRaises(ValueError):
                simulate(data, np.array([220]), np.array([1]), stop_distances=distances)
        with self.assertRaises(ValueError):
            simulate(data, np.array([220.5]), np.array([1]))
        with self.assertRaises(ValueError):
            StrategySpec("unknown", "fixed_1pct", 4)

    def test_stop_audit_uses_identical_entries_and_full_counterfactual_horizons(self):
        data = StrategyData.from_frame(market_frame(400))
        summary, records = stop_audit(data, data.entries(210, 390))
        self.assertEqual(summary["stopped_trades"], len(records))
        self.assertEqual(summary["baseline"]["stop_exits"], len(records))
        for record in records:
            self.assertAlmostEqual(record["fixed_1pct_net_return"], record["baseline_net_return"])
            self.assertLess(record["entry"] + 23, 390)

    def test_legacy_ledger_replay_and_assumption_mismatch(self):
        data = StrategyData.from_frame(market_frame(300))
        trade = simulate(data, np.array([220]), np.array([-1]), replace(data.config, slippage=0))["trades"][0]
        ledger = pd.DataFrame([
            dict(timestamp=data.frame.timestamp.iloc[220], action="OPEN_SHORT", price=trade["entry_fill"]),
            dict(timestamp=data.frame.timestamp.iloc[trade["exit"]], action="CLOSE_SHORT", price=trade["exit_fill"],
                 pnl=trade["net_pnl"], entry_notional=trade["entry_notional"], exit_reason=trade["reason"]),
        ])
        summary, records = audit_trade_ledger(data, ledger)
        self.assertTrue(summary["exact_replay"])
        self.assertEqual(summary["trades"], 1)
        self.assertAlmostEqual(records[0]["baseline_net_return"], trade["net_return"])
        ledger.loc[1, "pnl"] += 1
        with self.assertRaises(ValueError):
            audit_trade_ledger(data, ledger)

    def test_cli_writes_frozen_policy_and_preserves_existing_run(self):
        project = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "market.csv"
            market_frame(1000).to_csv(path, index=False)
            command = [sys.executable, str(project / "scripts/research_strategies.py"), "--data", str(path),
                       "--output-dir", str(root / "run"), "--folds", "2", "--min-trades", "2"]
            result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run/report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["protocol"]["variants"]), 48)
            for fold in report["folds"]:
                frozen = root / f"run/fold_{fold['fold']}/frozen_policy.json"
                self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), fold["frozen_policy_sha256"])
                policy = json.loads(frozen.read_text())
                self.assertEqual(policy["selected"], choose_on_selection(policy["selection"], 2))
                if not policy["enabled"]:
                    self.assertEqual(fold["policy_metrics"]["trades"], 0)
                self.assertFalse(policy["live_orders_allowed"])
                self.assertEqual(len(fold["evaluation"]), 48)
            again = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("new output directory", again.stderr)
