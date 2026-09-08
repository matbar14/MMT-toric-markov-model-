import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from toric_markov_model.train.checkpoint import dataset_from_checkpoint, load_checkpoint
from tests.support import market_frame


class PipelineTests(unittest.TestCase):
    def test_training_checkpoint_backtest_and_test_isolation(self):
        project = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with tempfile.TemporaryDirectory(prefix="toric-pipeline-") as directory:
            root = Path(directory)
            data = root / "synthetic.csv"
            market_frame().to_csv(data, index=False)
            training = [sys.executable, str(project / "scripts/train_trading_v3_basis.py"),
                        "--data", str(data), "--epochs", "1", "--seq-len", "8",
                        "--dim-angles", "16", "--num-states", "4", "--num-layers", "1",
                        "--dropout", "0", "--batch-size", "64", "--device", "cpu",
                        "--checkpoint-dir", str(root / "checkpoints")]
            result = subprocess.run(training, capture_output=True, text=True, env=environment, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint_path = root / "checkpoints/best_model_stage0.pt"
            checkpoint, model = load_checkpoint(checkpoint_path)
            dataset = dataset_from_checkpoint(checkpoint, data, split="test")
            prediction = model.detect_patterns(dataset[0][0][None])
            self.assertTrue(torch.isfinite(prediction["predicted_return"]).all())
            validation_end = checkpoint["split_boundaries"]["validation_end"]
            self.assertGreaterEqual(dataset.timestamps.iloc[0].isoformat(), validation_end)
            for stage in (1, 2):
                result = subprocess.run(training + ["--stage", str(stage), "--resume-from", str(checkpoint_path)],
                                         capture_output=True, text=True, env=environment, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((root / f"checkpoints/best_model_stage{stage}.pt").is_file())
            calibrated_path = root / "calibrated.pt"
            calibration = subprocess.run(
                [sys.executable, str(project / "scripts/calibrate_trading.py"),
                 "--data", str(data), "--checkpoint", str(checkpoint_path),
                 "--output", str(calibrated_path), "--min-signals", "2"],
                capture_output=True, text=True, env=environment, timeout=120,
            )
            self.assertEqual(calibration.returncode, 0, calibration.stdout + calibration.stderr)
            report = json.loads(calibrated_path.with_suffix(".json").read_text())
            self.assertFalse(report["test_used"])
            self.assertEqual(report["validation_partition"], checkpoint["validation_partition"])
            self.assertEqual(calibrated_path.exists(), report["accepted"])
            if report["accepted"]:
                calibrated, _ = load_checkpoint(calibrated_path)
                self.assertEqual(calibrated["decision_thresholds"], report["decision_thresholds"])
            unchanged, _ = load_checkpoint(checkpoint_path)
            self.assertNotIn("threshold_calibration", unchanged)
            command = [sys.executable, str(project / "scripts/backtest_trading_v3.py"),
                       "--data", str(data), "--checkpoint", str(checkpoint_path),
                       "--device", "cpu", "--output", str(root / "backtest.csv")]
            result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (root / "backtest_metrics.json").open() as stream:
                metrics = json.load(stream)
            self.assertIn("final_value", metrics)
            rejected = subprocess.run(command + ["--optimize-signal-threshold"],
                                       capture_output=True, text=True, env=environment, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("never on test", rejected.stderr)
            analysis = subprocess.run(
                [sys.executable, str(project / "scripts/analyze_confidence.py"),
                 "--data", str(data), "--checkpoint", str(checkpoint_path), "--samples", "2"],
                capture_output=True, text=True, env=environment, timeout=120,
            )
            self.assertEqual(analysis.returncode, 0, analysis.stdout + analysis.stderr)
