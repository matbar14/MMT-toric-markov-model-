import tempfile
import unittest
from pathlib import Path

import torch
from setuptools import find_packages

from toric_markov_model.train.checkpoint import FORMAT_VERSION, load_checkpoint, save_checkpoint
from tests.test_model import tiny_model


class CheckpointTests(unittest.TestCase):
    def test_package_is_discovered(self):
        packages = find_packages(where=str(Path(__file__).resolve().parents[1]))
        self.assertIn("toric_markov_model", packages)
        self.assertIn("toric_markov_model.model", packages)

    def test_strict_roundtrip_and_atomic_file_cleanup(self):
        model = tiny_model().eval()
        model.set_aux_target_stats({"aux_target_mean": [0.1] * 4, "aux_target_std": [0.2] * 4})
        features = torch.randn(1, 8, 28)
        expected = model.detect_patterns(features)
        with tempfile.TemporaryDirectory(prefix="toric-checkpoint-") as directory:
            path = Path(directory) / "model.pt"
            payload = dict(format_version=FORMAT_VERSION, model_config=model.config,
                           model_state_dict=model.state_dict())
            save_checkpoint(payload, path)
            _, loaded = load_checkpoint(path)
            actual = loaded.detect_patterns(features)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name])
            self.assertEqual(len(list(Path(directory).glob("*.tmp"))), 0)
            del payload["model_state_dict"]["pattern_head.0.weight"]
            save_checkpoint(payload, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path)

    def test_old_format_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="toric-checkpoint-") as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint({"format_version": 1}, path)
            with self.assertRaisesRegex(ValueError, "retrain"):
                load_checkpoint(path)

    def test_inference_defaults_use_saved_thresholds_and_allow_explicit_override(self):
        model = tiny_model(predict_return=False)
        thresholds = dict(confidence_threshold=0.0, pattern_prob_threshold=0.0, gate_threshold=1.0)
        payload = dict(format_version=FORMAT_VERSION, model_config=model.config,
                       model_state_dict=model.state_dict(), decision_thresholds=thresholds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(payload, path)
            _, loaded = load_checkpoint(path)
            features = torch.zeros(2, 8, 28)
            self.assertFalse(loaded.detect_patterns(features)["has_pattern"].any())
            self.assertTrue(loaded.detect_patterns(features, gate_threshold=0.0)["has_pattern"].all())
            payload["decision_thresholds"]["gate_threshold"] = float("nan")
            save_checkpoint(payload, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
