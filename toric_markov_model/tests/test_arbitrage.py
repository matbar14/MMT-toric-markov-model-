import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from tests.support import market_frame
from toric_markov_model.arbitrage import ArbitrageConfig, ArbitrageData, pair_outcome, simulate_pairs
from toric_markov_model.data.arbitrage_download import download_inputs, fetch_history, load_inputs
from toric_markov_model.train.arbitrage import fit_pair_models, forecasts, load_pair_checkpoint, save_pair_checkpoint, targets


def fixture(rows=700, config=None):
    market = market_frame(rows)
    for side in ("spot", "futures"):
        for field in ("open", "high", "low", "close"):
            market[f"{side}_{field}"] = 100.0
    marks = pd.DataFrame({"timestamp": market.timestamp, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    funding = pd.DataFrame(dict(timestamp=pd.date_range(market.timestamp.iloc[0] - pd.Timedelta(hours=12),
                                                       market.timestamp.iloc[-1] + pd.Timedelta(hours=13), freq="4h"),
                                funding_rate=0.0, mark_price=100.0))
    return ArbitrageData.from_frames(market, funding, marks, config)


def write_inputs(root, data):
    root.mkdir(parents=True, exist_ok=True)
    market_path = root / "market.csv"
    data.frame.to_csv(market_path, index=False)
    market_hash = hashlib.sha256(market_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(dict(symbol="BTCUSDT", sha256=market_hash)))
    output = root / "arbitrage"
    output.mkdir()
    files = {}
    for name, frame in (("funding.csv", data.funding), ("marks.csv", data.marks)):
        frame.to_csv(output / name, index=False)
        files[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(dict(format="arbitrage_inputs_v1", symbol="BTCUSDT",
                                                          market_sha256=market_hash, files=files)))
    return market_path, output


class Client:
    def __init__(self, batches):
        self.batches, self.calls = iter(batches), []

    def get(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        self.batch = next(self.batches)
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return self.batch


class ArbitrageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_equal_btc_quantities_cancel_common_price_move(self):
        config = ArbitrageConfig(spot_fee=0, futures_fee=0, spot_slippage=0, futures_slippage=0)
        data = fixture(config=config)
        prices = 100 + np.arange(len(data.frame)) * 0.1
        for side in ("spot", "futures"):
            for field in ("open", "high", "low", "close"):
                data.frame[f"{side}_{field}"] = prices
        for field in ("open", "high", "low", "close"):
            data.marks[field] = prices
        outcome = pair_outcome(data, 103)
        self.assertGreater(outcome["spot_pnl"], 0)
        self.assertAlmostEqual(outcome["spot_pnl"], -outcome["futures_pnl"])
        self.assertEqual(outcome["net_pnl"], 0)
        run = simulate_pairs(data, np.array([103, 110]), np.array([1, 1]))
        np.testing.assert_allclose(run["equity"], 1)
        for trade in run["trades"]:
            self.assertEqual(trade["spot_quantity"], -trade["futures_quantity"])
            self.assertLessEqual(trade["committed_capital"], config.allocation + 1e-12)

    def test_basis_convergence_and_four_fill_costs(self):
        data = fixture()
        data.frame.loc[103, "futures_open"] = 102.0
        outcome = pair_outcome(data, 103)
        expected_spot = 100 * (1 - data.config.spot_slippage) - 100 * (1 + data.config.spot_slippage)
        expected_future = 102 * (1 - data.config.futures_slippage) - 100 * (1 + data.config.futures_slippage)
        expected_fees = ((outcome["spot_entry"] + outcome["spot_exit"]) * data.config.spot_fee +
                         (outcome["futures_entry"] + outcome["futures_exit"]) * data.config.futures_fee)
        self.assertAlmostEqual(outcome["net_pnl"], expected_spot + expected_future - expected_fees)
        self.assertAlmostEqual(outcome["net_return"], outcome["net_pnl"] / outcome["committed_per_unit"])
        self.assertEqual(outcome["spot_quantity"], -outcome["futures_quantity"])
        self.assertLess(pair_outcome(data, 103, data.config.stressed())["net_pnl"], outcome["net_pnl"])

    def test_funding_uses_events_mark_price_and_short_sign(self):
        data = fixture()
        event = data.frame.timestamp.iloc[104]
        data.funding.loc[data.funding.timestamp == event, ["funding_rate", "mark_price"]] = [0.001, 120]
        outcome = pair_outcome(data, 103)
        self.assertAlmostEqual(outcome["funding_pnl"], 0.12)
        data.funding.loc[data.funding.timestamp == event, "funding_rate"] = -0.001
        self.assertAlmostEqual(pair_outcome(data, 103)["funding_pnl"], -0.12)

    def test_funding_boundary_is_conservative_and_no_overnight(self):
        data = fixture()
        boundary = data.frame.timestamp.iloc[103]
        added = pd.DataFrame(dict(timestamp=[boundary, boundary + pd.Timedelta(hours=6)],
                                  funding_rate=[0.001, 0.001], mark_price=[100, 100]))
        data.funding = pd.concat([data.funding, added]).sort_values("timestamp").reset_index(drop=True)
        self.assertEqual(pair_outcome(data, 103)["funding_pnl"], 0)
        data.funding.loc[data.funding.timestamp.isin(added.timestamp), "funding_rate"] = -0.001
        self.assertAlmostEqual(pair_outcome(data, 103)["funding_pnl"], -0.2)
        entries = data.entries(0, len(data.frame))
        run = simulate_pairs(data, entries, np.ones(len(entries), dtype=int))
        self.assertEqual(run["metrics"]["signals"], run["metrics"]["trades"])
        for trade in run["trades"]:
            self.assertEqual(pd.Timestamp(trade["entry_time"]).date(), pd.Timestamp(trade["exit_time"]).date())

    def test_short_wallet_can_fail_despite_delta_neutrality(self):
        data = fixture()
        data.marks.loc[104, "high"] = 250
        with self.assertRaisesRegex(ValueError, "collateral risk"):
            pair_outcome(data, 103)

    def test_future_settlements_and_entry_candles_are_not_features(self):
        data = fixture()
        changed_market, changed_funding, changed_marks = data.frame.copy(), data.funding.copy(), data.marks.copy()
        for side in ("spot", "futures"):
            for field in ("open", "high", "low", "close"):
                changed_market.loc[103:, f"{side}_{field}"] *= 1.1
        changed_funding.loc[changed_funding.timestamp >= changed_market.timestamp.iloc[103], "funding_rate"] = 0.02
        changed = ArbitrageData.from_frames(changed_market, changed_funding, changed_marks)
        np.testing.assert_array_equal(data.windows([103]), changed.windows([103]))
        np.testing.assert_array_equal(data.baseline_actions([103]), changed.baseline_actions([103]))
        np.testing.assert_array_equal(data.carry_actions([103]), changed.carry_actions([103]))

    def test_missing_funding_marks_and_invalid_schedules_rejected(self):
        data = fixture()
        for funding in (data.funding.iloc[:0], data.funding.iloc[:-4], data.funding.drop(index=[3, 4])):
            with self.assertRaises(ValueError):
                ArbitrageData.from_frames(data.frame, funding, data.marks)
        with self.assertRaises(ValueError):
            ArbitrageData.from_frames(data.frame, data.funding, data.marks.iloc[1:])
        for entries, actions in (([97], [1]), ([103, 104], [1, 1]), ([103], [-1])):
            with self.assertRaises(ValueError):
                simulate_pairs(data, entries, actions)

    def test_flat_prices_lose_costs_no_fake_positive_baseline(self):
        data = fixture()
        entries = data.entries(0, len(data.frame))
        self.assertFalse(data.baseline_actions(entries).any())
        self.assertFalse(data.carry_actions(entries).any())
        raw = simulate_pairs(data, entries, np.ones(len(entries), dtype=int))
        self.assertLess(raw["metrics"]["total_return_pct"], 0)
        self.assertEqual(raw["metrics"]["funding_fraction"], 0)
        self.assertAlmostEqual(np.prod(1 + raw["daily_returns"]), raw["equity"][-1])
        empty = simulate_pairs(data, entries, np.zeros(len(entries), dtype=int))
        np.testing.assert_array_equal(empty["equity"], 1)

    def test_intraday_slots_can_capture_eight_hour_settlements(self):
        data = fixture()
        events = data.funding.iloc[::2].copy()
        events.timestamp += pd.Timedelta(hours=4)
        events.funding_rate = 0.001
        data = ArbitrageData.from_frames(data.frame, events, data.marks)
        for entry in (103, 110):
            self.assertAlmostEqual(pair_outcome(data, entry)["funding_pnl"], 0.1)

    def test_pair_model_training_checkpoint_and_legacy_rejection(self):
        data = fixture()
        train, selection = data.entries(0, 400), data.entries(400, 550)
        with self.assertRaisesRegex(ValueError, "precede"):
            fit_pair_models(data, train, train, epochs=1)
        model, ridge, history = fit_pair_models(data, train, selection, epochs=2)
        prediction = forecasts(model, ridge, data.windows(selection))
        self.assertTrue(np.isfinite(prediction["toric"]).all())
        self.assertAlmostEqual(history["train_mean"], float(targets(data, train).mean()), places=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.pt"
            save_pair_checkpoint(model, data, path, dict(enabled=False), dict(synthetic=True))
            checkpoint, restored, config = load_pair_checkpoint(path)
            self.assertEqual(config, data.config)
            np.testing.assert_array_equal(forecasts(restored, ridge, data.windows(selection))["toric"], prediction["toric"])
            checkpoint["format"] = "toric_payoff_v1"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "research arbitrage"):
                load_pair_checkpoint(path)

    def test_input_checksums_and_end_to_end_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_path, inputs = write_inputs(root, fixture(2200))
            loaded, _ = load_inputs(market_path, inputs)
            self.assertEqual(len(loaded.frame), 2200)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/research_arbitrage.py"),
                       "--market-data", str(market_path), "--arbitrage-data", str(inputs),
                       "--output-dir", str(root / "run"), "--folds", "2", "--epochs", "1", "--min-trades", "5"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run/report.json").read_text())
            self.assertTrue(report["complete"])
            for fold in report["folds"]:
                self.assertFalse(fold["enabled"])
                self.assertEqual(fold["frozen_policy"]["trades"], 0)
                frozen = root / f"run/fold_{fold['fold']}/frozen_policy.json"
                self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(), fold["frozen_policy_sha256"])
                for name, digest in json.loads(frozen.read_text())["weights_sha256"].items():
                    self.assertEqual(hashlib.sha256((frozen.parent / name).read_bytes()).hexdigest(), digest)
            again = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("new output directory", again.stderr)
            with (inputs / "funding.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_inputs(market_path, inputs)

    def test_positive_synthetic_carry_does_not_credit_model_for_trivial_funding(self):
        data = fixture(2200)
        data.funding["funding_rate"] = 0.01
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_path, inputs = write_inputs(root, data)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/research_arbitrage.py"),
                       "--market-data", str(market_path), "--arbitrage-data", str(inputs),
                       "--output-dir", str(root / "run"), "--folds", "2", "--epochs", "1", "--min-trades", "5"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run/report.json").read_text())
            for fold in report["folds"]:
                self.assertTrue(fold["enabled"])
                self.assertEqual(fold["selected"], "carry")
                self.assertEqual(fold["reference"], "carry")
                self.assertGreater(fold["frozen_policy"]["total_return_pct"], 0)
                self.assertEqual(fold["diagnostic"]["baseline"]["trades"], 0)
                checkpoint, _, _ = load_pair_checkpoint(root / f"run/fold_{fold['fold']}/toric_arbitrage.pt")
                self.assertFalse(checkpoint["live_orders_allowed"])

    def test_mixed_timestamp_precision_is_normalized(self):
        data = fixture()
        events = data.funding.copy()
        events.loc[events.index[::2], "timestamp"] += pd.Timedelta(milliseconds=1)
        events["timestamp"] = events.timestamp.astype(str)
        rebuilt = ArbitrageData.from_frames(data.frame, events, data.marks)
        self.assertEqual(str(rebuilt.funding.timestamp.dtype), "datetime64[ns, UTC]")

    @patch("toric_markov_model.data.arbitrage_download.time.sleep")
    def test_download_pagination_resume_and_mark_zero_volume(self, sleep):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            market_path, _ = write_inputs(root / "source", fixture(48))
            start = int(pd.Timestamp("2025-01-01T00:00Z").timestamp() * 1000)
            hour = 3_600_000
            marks = [[start + offset * hour, "100", "100", "100", "100", "0",
                      start + (offset + 1) * hour - 1, "0", 0, "0", "0", "0"] for offset in range(48)]
            funding = [dict(symbol="BTCUSDT", fundingTime=start + offset * hour,
                            fundingRate="0.0001", markPrice="100") for offset in range(-12, 60, 4)]
            output = root / "download"
            with self.assertRaises(StopIteration):
                download_inputs(Client([marks[:24]]), market_path, output)
            client = Client([marks[24:], funding[:8], funding[8:], []])
            manifest = download_inputs(client, market_path, output, resume=True)
            self.assertEqual(client.calls[0][1]["params"]["startTime"], start + 24 * hour)
            self.assertEqual(manifest["format"], "arbitrage_inputs_v1")
            data, _ = load_inputs(market_path, output)
            self.assertEqual(len(data.marks), 48)
            with self.assertRaisesRegex(ValueError, "completed"):
                download_inputs(Client([]), market_path, output, resume=True)
            with self.assertRaisesRegex(ValueError, "different market"):
                fetch_history(Client([]), "marks", "OTHER", start, start + 48 * hour, output / "marks_checkpoint.json")
