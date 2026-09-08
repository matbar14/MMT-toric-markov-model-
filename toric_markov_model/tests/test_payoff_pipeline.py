from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from tests.support import market_frame
from toric_markov_model.data.payoff import PayoffData
from toric_markov_model.model.payoff import ToricPayoffModel
from toric_markov_model.train.checkpoint import save_checkpoint
from toric_markov_model.train.payoff_checkpoint import forecast_closed_bars, load_payoff_checkpoint


class PayoffPipelineTests(unittest.TestCase):
    def test_safe_load_and_rejected_policy_remains_hold(self):
        torch.set_num_threads(1)
        data = PayoffData.from_frame(market_frame(500))
        model = ToricPayoffModel(len(data.feature_names), max_len=8, dim_angles=16, num_states=8)
        model.fit_statistics(data.features[100:200], data.net_returns[108:200])
        payload = dict(format="toric_payoff_v1", model_config=model.config,
                       model_state_dict=model.state_dict(), feature_names=data.feature_names,
                       execution=asdict(data.config), toric_policy=dict(enabled=False, pattern_filter=False,
                       threshold=0.0, execution=asdict(data.config)), selected_model="toric",
                       dates=dict(calibration=dict(last_target=data.frame.timestamp.iloc[300].isoformat()),
                                  train=dict(first_context=data.frame.timestamp.iloc[100].isoformat(),
                                             first_entry=data.frame.timestamp.iloc[108].isoformat())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payoff.pt"
            save_checkpoint(payload, path)
            loaded, restored, _ = load_payoff_checkpoint(path)
            forecast = forecast_closed_bars(loaded, restored, data.frame)
            self.assertEqual(forecast["paper_action"], "HOLD")
            self.assertFalse(forecast["live_orders_allowed"])
            self.assertFalse(forecast["production_approved"])
            with self.assertRaises(ValueError):
                forecast_closed_bars(loaded, restored, data.frame.iloc[:200])
            with self.assertRaises(ValueError):
                forecast_closed_bars(loaded, restored, data.frame.iloc[::2])
            payload["model_state_dict"]["feature_std"].fill_(0)
            save_checkpoint(payload, path)
            with self.assertRaises(ValueError):
                load_payoff_checkpoint(path)

    def test_research_cli_smoke_and_new_directory_guard(self):
        project = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "market.csv"
            market_frame(1000).to_csv(data, index=False)
            command = [sys.executable, str(project / "scripts/research_payoff.py"),
                       "--data", str(data), "--output-dir", str(root / "run"),
                       "--epochs", "1", "--folds", "2", "--seq-len", "8", "--min-trades", "2"]
            result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run/report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["folds"]), 2)
            self.assertFalse(report["protocol"]["production_approved"])
            checkpoint = root / "run/fold_2/toric_payoff.pt"
            inference = subprocess.run([sys.executable, str(project / "scripts/predict_payoff.py"),
                                        "--data", str(data), "--checkpoint", str(checkpoint)],
                                       capture_output=True, text=True, env=environment, timeout=30)
            self.assertEqual(inference.returncode, 0, inference.stdout + inference.stderr)
            forecast = json.loads(inference.stdout)
            self.assertFalse(forecast["live_orders_allowed"])
            rejected = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("new output directory", rejected.stderr)
