import importlib.util
import unittest
from pathlib import Path

import numpy as np


script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_trading.py"
spec = importlib.util.spec_from_file_location("diagnostics_script", script)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class SampleWindows:
    seq_len = 3

    def __init__(self):
        self.features = np.arange(30, dtype=np.float32).reshape(10, 3)

    def __len__(self):
        return 5


class DiagnosticTests(unittest.TestCase):
    def test_baseline_windows_match_dataset_context_without_future_rows(self):
        dataset = SampleWindows()
        actual = diagnostics.tabular_features(dataset)
        expected = []
        for index in range(len(dataset)):
            window = dataset.features[index:index + dataset.seq_len]
            expected.append(np.concatenate((window[-1], window.mean(0), window.std(0), window[-1] - window[0])))
        np.testing.assert_allclose(actual, expected)
        dataset.features[7:] = 1000
        np.testing.assert_array_equal(diagnostics.tabular_features(dataset), actual)

    def test_constant_scores_have_prevalence_average_precision(self):
        labels = np.array([True, False, False, True])
        metrics = diagnostics.ranking_metrics(labels, np.full(4, 0.3))
        self.assertEqual(metrics["average_precision"], metrics["prevalence"])
        self.assertEqual(metrics["roc_auc"], 0.5)
        metrics = diagnostics.ranking_metrics(np.zeros(4, dtype=bool), np.zeros(4))
        self.assertIsNone(metrics["average_precision"])
        self.assertIsNone(metrics["roc_auc"])
