import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.train.checkpoint import load_checkpoint
from toric_markov_model.train.cpp_bridge import archive_name, export_bundle, import_weights
from toric_markov_model.train.trading import compute_loss, configure_stage_trainability, run_epoch
from toric_markov_model.train.selection import partition_validation
from tests.support import market_frame
from tests.test_model import tiny_model


PROJECT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("TORIC_CPP_BINARY", PROJECT / "cpp/build/toric_train"))


@unittest.skipUnless(BINARY.is_file(), "build cpp/toric_train with CMake to run native parity tests")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory(prefix="toric-cpp-test-")
        cls.root = Path(cls.directory.name)
        cls.csv = cls.root / "market.csv"
        market_frame(400).to_csv(cls.csv, index=False)
        config = dict(seq_len=8, prediction_horizon=4, train_split=0.7,
                      validation_split=0.15, verbose=False, return_aux_targets=True)
        cls.train = TradingDatasetV3(cls.csv, **config, split="train")
        cls.validation = TradingDatasetV3(
            cls.csv, **config, split="validation", normalization_stats=cls.train.get_normalization_stats(),
            aux_target_stats=cls.train.get_aux_target_stats(), split_boundaries=cls.train.split_boundaries,
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_native_forward_gradients_and_adamw_match_python(self):
        for stage in (0, 1, 2):
            with self.subTest(stage=stage):
                torch.manual_seed(42)
                model = tiny_model()
                model.set_aux_target_stats(self.train.get_aux_target_stats())
                folder = self.root / f"parity-{stage}"
                folder.mkdir()
                bundle_path = folder / "input.pt"
                archive = export_bundle(model, self.train, self.validation, bundle_path, stage=stage)
                result = subprocess.run(
                    [str(BINARY), "--input", str(bundle_path), "--output-dir", str(folder), "--check",
                     "--stage", str(stage), "--batch-size", "16", "--epochs", "1", "--threads", "1"],
                    capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                diagnostic = torch.jit.load(str(folder / "diagnostic.pt"))
                configure_stage_trainability(model, stage)
                model.train()
                features, labels, auxiliary = [torch.stack([self.train[index][column] for index in range(16)])
                                               for column in range(3)]
                outputs = model(features)
                torch.testing.assert_close(outputs["pattern_logits"], diagnostic.pattern, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(outputs["non_hold_logit"], diagnostic.gate, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(torch.cat([outputs[name] for name in model.AUX_NAMES], 1),
                                           diagnostic.auxiliary, rtol=2e-5, atol=2e-6)
                loss, _ = compute_loss(model, outputs, labels, auxiliary,
                                       archive.positive_weight, archive.gate_weight, stage=stage)
                torch.testing.assert_close(loss, diagnostic.loss, rtol=2e-5, atol=2e-6)
                loss.backward()
                buffers = dict(diagnostic.named_buffers())
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, buffers["gradient__" + archive_name(name)],
                                                   rtol=5e-4, atol=5e-6, msg=name)
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=5e-5)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                native_model = tiny_model()
                import_weights(native_model, folder / "last_weights.pt")
                native_parameters = dict(native_model.named_parameters())
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, native_parameters[name], rtol=2e-5, atol=2e-6, msg=name)

    def test_cli_exports_compatible_checkpoint_and_keeps_partial_batch(self):
        folder = self.root / "cli"
        command = [sys.executable, str(PROJECT / "scripts/train_trading_cpp.py"), "--data", str(self.csv),
                   "--checkpoint-dir", str(folder), "--binary", str(BINARY), "--epochs", "3",
                   "--seq-len", "8", "--dim-angles", "16", "--num-states", "4", "--num-layers", "1",
                   "--batch-size", "32", "--dropout", "0.2", "--threads", "1"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        checkpoint, model = load_checkpoint(folder / "best_model_stage0.pt")
        self.assertEqual(checkpoint["backend"], "libtorch-cpp")
        selection, _, partition = partition_validation(self.validation)
        self.assertEqual(checkpoint["validation_metrics"]["samples"], len(selection))
        self.assertEqual(checkpoint["validation_partition"], partition)
        self.assertEqual(checkpoint["feature_names"], self.train.feature_names)
        prediction = model.detect_patterns(self.validation[0][0][None])
        self.assertTrue(torch.isfinite(prediction["predicted_return"]).all())
        with (folder / "native/metrics.jsonl").open() as stream:
            records = [json.loads(line) for line in stream]
        best_loss, expected_epoch = float("inf"), None
        for record in records:
            validation = record["validation"]
            loss = validation["pattern_loss"] + validation["gate_loss"] + 0.01 * validation["aux_loss"]
            if loss < best_loss - 1e-4:
                best_loss, expected_epoch = loss, record["epoch"]
        self.assertEqual(checkpoint["epoch"], expected_epoch)
        self.assertEqual(checkpoint["selection_metric"], "loss")
        self.assertEqual(checkpoint["selection_mode"], "min")
        self.assertEqual(record["train"]["samples"], len(self.train))
        self.assertEqual(record["train"]["optimizer_updates"], (len(self.train) + 31) // 32)
        inputs = torch.jit.load(str(folder / "native/input.pt"))
        self.assertFalse(any("test_" in name for name, _ in inputs.named_buffers()))
        metrics = run_epoch(model, torch.utils.data.DataLoader(selection, batch_size=32), "cpu",
                            inputs.positive_weight, inputs.gate_weight)
        for name in ("loss", "pattern_loss", "gate_loss", "aux_loss", "pattern_f1", "gate_f1"):
            self.assertAlmostEqual(metrics[name], checkpoint["validation_metrics"][name], places=5, msg=name)

    def test_hold_only_stage2_does_not_update(self):
        model = tiny_model()
        model.set_aux_target_stats(self.train.get_aux_target_stats())
        folder = self.root / "hold-only"
        folder.mkdir()
        archive = export_bundle(model, self.train, self.validation, folder / "input.pt", stage=2)
        with torch.no_grad():
            archive.train_labels[:16].zero_()
            archive.train_labels[:16, -1] = 1
        archive.save(str(folder / "input.pt"))
        result = subprocess.run([str(BINARY), "--input", str(folder / "input.pt"), "--output-dir", str(folder),
                                 "--check", "--stage", "2", "--batch-size", "16", "--weight-decay", "0.5"],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        native = tiny_model()
        import_weights(native, folder / "last_weights.pt")
        for actual, expected in zip(native.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_arguments_fail(self):
        result = subprocess.run([str(BINARY), "--input", "missing", "--output-dir", str(self.root),
                                 "--batch-size", "0"], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive", result.stderr)
