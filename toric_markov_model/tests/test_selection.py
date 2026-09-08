import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.support import market_frame
from tests.test_model import tiny_model
from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.train.calibration import decision_metrics, fit_thresholds
from toric_markov_model.train.selection import improves_loss, partition_validation


class SelectionTests(unittest.TestCase):
    def test_validation_partitions_have_disjoint_windows_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "market.csv"
            market_frame().to_csv(path, index=False)
            train = TradingDatasetV3(path, seq_len=8, prediction_horizon=4, split="train")
            validation = TradingDatasetV3(
                path, seq_len=8, prediction_horizon=4, split="validation",
                split_boundaries=train.split_boundaries, normalization_stats=train.get_normalization_stats(),
                aux_target_stats=train.get_aux_target_stats(),
            )
            selection, calibration, metadata = partition_validation(validation)
            self.assertLess(selection.timestamps.iloc[-1], calibration.timestamps.iloc[0])
            self.assertEqual(metadata["selection_samples"], len(selection))
            self.assertEqual(metadata["calibration_samples"], len(calibration))
            np.testing.assert_array_equal(selection.patterns, validation.patterns[:len(selection.features)])
            np.testing.assert_array_equal(calibration.features, validation.features[len(selection.features):])
            self.assertEqual(len(selection) + len(calibration), len(validation) - 11)
            with self.assertRaises(ValueError):
                partition_validation(train)
            with self.assertRaises(ValueError):
                partition_validation(validation, 0.99)

    def test_loss_improvement_does_not_require_any_thresholded_signals(self):
        self.assertTrue(improves_loss(0.7, float("inf")))
        self.assertTrue(improves_loss(0.6, 0.7))
        self.assertFalse(improves_loss(0.70001, 0.7))
        with self.assertRaises(ValueError):
            improves_loss(float("nan"), 0.7)

    def test_threshold_fitting_can_recover_events_below_half(self):
        labels = np.zeros((100, 16))
        labels[:20, 0] = 1
        scores = np.full_like(labels, 0.01)
        scores[:20, 0] = 0.4
        report = fit_thresholds(scores, np.full(100, 0.4), labels)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["default_metrics"]["signals"], 0)
        self.assertEqual(report["selected_metrics"]["strongest_f1"], 1)

    def test_uninformative_scores_and_no_events_are_not_forced_to_trade(self):
        labels = np.zeros((100, 16))
        for events in (0, 20):
            labels[:events, 0] = 1
            self.assertFalse(fit_thresholds(np.full_like(labels, 0.4), np.full(100, 0.4), labels)["accepted"])

    def test_nearly_always_active_candidate_is_rejected(self):
        labels = np.zeros((100, 16))
        labels[:10, 0] = 1
        labels[10:20, 1] = 1
        scores = np.full_like(labels, 0.01)
        scores[:10, 0] = 0.4
        scores[10:99, 1] = 0.4
        report = fit_thresholds(scores, np.full(100, 0.4), labels)
        self.assertFalse(report["accepted"])
        self.assertTrue(any(candidate["metrics"]["strongest_f1"] > report["always_most_common_pattern_f1"]
                            for candidate in report["candidates"]))

    def test_metrics_match_model_decoding(self):
        model = tiny_model()
        outputs = {"pattern_logits": torch.randn(40, 17), "non_hold_logit": torch.randn(40, 1)}
        labels = torch.rand(40, 16) > 0.9
        thresholds = dict(pattern_prob_threshold=0.4, gate_threshold=0.3, confidence_threshold=0.2)
        decoded = model.decode_outputs(outputs, **thresholds)
        metrics = decision_metrics(outputs["pattern_logits"][:, :-1].sigmoid().numpy(),
                                   outputs["non_hold_logit"].sigmoid().numpy(), labels.numpy(), thresholds)
        self.assertEqual(metrics["signals"], decoded["has_pattern"].sum().item())
        best = decoded["best_non_hold_pattern"]
        correct = labels[torch.arange(40), best] & decoded["has_pattern"]
        self.assertEqual(metrics["correct_patterns"], correct.sum().item())

    def test_invalid_calibration_inputs_are_rejected(self):
        for conditional, gate, labels in (
            (np.zeros((0, 16)), np.zeros(0), np.zeros((0, 16))),
            (np.zeros((2, 16)), np.zeros(3), np.zeros((2, 16))),
            (np.full((2, 16), np.nan), np.zeros(2), np.zeros((2, 16))),
            (np.zeros((2, 16)), np.full(2, 1.1), np.zeros((2, 16))),
            (np.zeros((2, 16)), np.zeros(2), np.full((2, 16), np.nan)),
        ):
            with self.assertRaises(ValueError):
                fit_thresholds(conditional, gate, labels)
