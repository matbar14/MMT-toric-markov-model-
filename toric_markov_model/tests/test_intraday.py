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
import torch

from tests.support import market_frame
from toric_markov_model.model.payoff import ToricPayoffModel
from toric_markov_model.signal_filter import (
    ENTRY_HOURS_UTC, SIGNALS, SignalData, base_screen, filter_actions,
    paired_comparison, predict_filters, run_signal, select_signal, train_entry_filters,
)


class IntradayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_entries_are_intraday_and_execution_matches_training_targets(self):
        data = SignalData.from_frame(market_frame(400))
        entries = data.entries(0, 400, 32)
        self.assertEqual(set(data.frame.timestamp.iloc[entries].dt.hour), set(ENTRY_HOURS_UTC))
        run = run_signal(data, entries, np.ones(len(entries), dtype=int))
        self.assertEqual(run["metrics"]["trades"], len(entries))
        for trade in run["trades"]:
            self.assertLess(trade["exit"] - trade["entry"], 6)
            self.assertEqual(data.frame.timestamp.iloc[trade["entry"]].date(),
                             data.frame.timestamp.iloc[trade["exit"]].date())
            self.assertAlmostEqual(trade["net_return"], data.net_returns[trade["entry"], 0], places=12)
            self.assertEqual(trade["exit"], data.exit_indices[trade["entry"], 0])

    def test_six_hour_timeout_including_last_slot_and_costs(self):
        frame = market_frame(300)
        for market in ("spot", "futures"):
            for field in ("open", "high", "low", "close"):
                frame[f"{market}_{field}"] = 100.0
        data = SignalData.from_frame(frame)
        entries = data.entries(0, 300, 32)
        run = run_signal(data, entries, np.ones(len(entries), dtype=int))
        self.assertLess(run["metrics"]["total_return_pct"], 0)
        for trade in run["trades"]:
            self.assertEqual(trade["exit"], trade["entry"] + 5)
            self.assertEqual(trade["reason"], "MAX_HOLD")
        stress = run_signal(data, entries, np.ones(len(entries), dtype=int), stress=True)
        self.assertLess(stress["metrics"]["total_return_pct"], run["metrics"]["total_return_pct"])

    def test_entry_bar_and_future_are_not_signal_inputs(self):
        frame = market_frame(400)
        original = SignalData.from_frame(frame)
        for market in ("spot", "futures"):
            for field in ("open", "high", "low", "close"):
                frame.loc[240:, f"{market}_{field}"] *= 2
        changed = SignalData.from_frame(frame)
        entries = np.array([240])
        np.testing.assert_array_equal(original.windows(entries, 32), changed.windows(entries, 32))
        for name in SIGNALS:
            np.testing.assert_array_equal(original.actions(name, entries), changed.actions(name, entries))

    def test_invalid_cadence_slots_and_short_actions_rejected(self):
        frame = market_frame(300)
        shifted = frame.copy()
        shifted.timestamp += pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError):
            SignalData.from_frame(shifted)
        data = SignalData.from_frame(frame)
        for entries, actions in (([241], [1]), ([240, 241], [1, 1]), ([240], [-1]), ([298], [1])):
            with self.assertRaises(ValueError):
                run_signal(data, np.array(entries), np.array(actions))

    def test_weekly_frequency_and_abstention_cannot_pass(self):
        positive = dict(trades=40, trades_per_day=1, total_return_pct=1, block_lower_daily_mean=0.001,
                        max_drawdown_pct=-0.5)
        sparse = {**positive, "trades_per_day": 1 / 7}
        self.assertIsNone(select_signal({name: sparse for name in SIGNALS}))
        self.assertFalse(base_screen({"metrics": sparse}, {"metrics": positive}))
        dates = pd.date_range("2025-01-01", periods=30).astype(str).tolist()
        base = dict(metrics=positive, daily_returns=np.ones(30) * 0.01, daily_dates=dates)
        empty = dict(metrics={**positive, "trades": 0, "trades_per_day": 0}, daily_returns=np.zeros(30), daily_dates=dates)
        self.assertFalse(paired_comparison(base, empty, base, empty)["improved"])
        with self.assertRaisesRegex(ValueError, "identical daily"):
            paired_comparison(base, {**empty, "daily_dates": dates[::-1]}, base, empty)
        np.testing.assert_array_equal(filter_actions([1, 0, 1], [0.1, 0.2, -0.1]), [1, 0, 0])

    def test_long_only_filter_training_and_train_only_statistics(self):
        data = SignalData.from_frame(market_frame(700))
        train = data.entries(0, 420, 32)
        selection = data.entries(420, 580, 32)
        evaluation = data.entries(580, 700, 32)
        with self.assertRaisesRegex(ValueError, "chronologically disjoint"):
            train_entry_filters(data, train, train, epochs=1, seeds=(42,))
        models, ridge, histories = train_entry_filters(data, train, selection, epochs=1, seeds=(42,))
        forecasts, members = predict_filters(models, ridge, data.windows(evaluation, 32))
        self.assertEqual(histories[0]["best_epoch"], 1)
        self.assertEqual(models[0].num_outputs, 1)
        for values in forecasts.values():
            self.assertEqual(values.shape, evaluation.shape)
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(forecasts["toric"], members[0])
        np.testing.assert_allclose(models[0].target_mean.numpy(), data.net_returns[train, :1].mean(0), rtol=1e-5)
        restored = ToricPayoffModel(**models[0].config)
        restored.load_state_dict(models[0].state_dict(), strict=True)
        self.assertEqual(ToricPayoffModel(data.features.shape[1]).num_outputs, 2)

    def test_cli_freezes_policy_and_never_trains_on_failed_baseline(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = market_frame(2200)
            for market in ("spot", "futures"):
                for field in ("open", "high", "low", "close"):
                    frame[f"{market}_{field}"] = 100.0
            frame.to_csv(root / "market.csv", index=False)
            command = [sys.executable, str(project / "scripts/research_intraday.py"), "--data", str(root / "market.csv"),
                       "--output-dir", str(root / "run"), "--folds", "2", "--epochs", "1"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run/report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["protocol"]["execution"]["horizon"], 6)
            for fold in report["folds"]:
                frozen = root / f"run/fold_{fold['fold']}/frozen_policy.json"
                self.assertEqual(fold["frozen_policy_sha256"], hashlib.sha256(frozen.read_bytes()).hexdigest())
                self.assertFalse(fold["base_enabled"])
                self.assertEqual(fold["filters"], {})
                self.assertEqual(fold["frozen_base_policy"]["trades"], 0)
            again = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("new output directory", again.stderr)

    def test_cli_positive_synthetic_baseline_exercises_filter_branch(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = market_frame(2200)
            prices = 100 * np.exp(np.arange(len(frame)) * 0.002)
            for market in ("spot", "futures"):
                frame[f"{market}_open"] = prices
                frame[f"{market}_high"] = prices * 1.002
                frame[f"{market}_low"] = prices
                frame[f"{market}_close"] = prices * 1.002
            frame.to_csv(root / "market.csv", index=False)
            result = subprocess.run(
                [sys.executable, str(project / "scripts/research_intraday.py"), "--data", str(root / "market.csv"),
                 "--output-dir", str(root / "run"), "--folds", "2", "--epochs", "1"],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run/report.json").read_text())
            for fold in report["folds"]:
                self.assertTrue(fold["base_enabled"])
                self.assertEqual(set(fold["filters"]), {"toric", "ridge"})
                folder = root / f"run/fold_{fold['fold']}"
                frozen = json.loads((folder / "frozen_policy.json").read_text())
                for name, digest in frozen["weights_sha256"].items():
                    self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest)
