import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from tests.support import market_frame


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="toric-data-test-")
        cls.path = Path(cls.directory.name) / "market.csv"
        cls.frame = market_frame()
        cls.frame.to_csv(cls.path, index=False)
        cls.config = dict(seq_len=8, prediction_horizon=4, train_split=0.7,
                          validation_split=0.15, verbose=False)
        cls.train = TradingDatasetV3(cls.path, **cls.config, return_aux_targets=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def evaluation(self, split):
        return TradingDatasetV3(
            self.path, **self.config, split=split, split_boundaries=self.train.split_boundaries,
            normalization_stats=self.train.get_normalization_stats(),
            aux_target_stats=self.train.get_aux_target_stats(),
        )

    def test_last_sample_has_complete_future(self):
        dataset = self.train
        features, _, auxiliary = dataset[len(dataset) - 1]
        entry = len(dataset) - 1 + dataset.seq_len
        exit_bar = entry + dataset.prediction_horizon - 1
        self.assertEqual(exit_bar, len(dataset.spot_close) - 1)
        physical = auxiliary.numpy() * dataset.aux_target_std + dataset.aux_target_mean
        expected = dataset.spot_close.iloc[exit_bar] / dataset.spot_open.iloc[entry] - 1
        self.assertAlmostEqual(float(physical[0]), expected, places=6)
        self.assertTrue(np.isfinite(physical).all())
        np.testing.assert_array_equal(features[-1].numpy(), dataset.features[entry - 1])
        self.assertGreater(abs(float(physical[1])), 1e-5)
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_disjoint_splits_and_train_only_statistics(self):
        validation, test = self.evaluation("validation"), self.evaluation("test")
        self.assertLess(self.train.timestamps.iloc[-1], validation.timestamps.iloc[0])
        self.assertLess(validation.timestamps.iloc[-1], test.timestamps.iloc[0])
        for dataset in (validation, test):
            np.testing.assert_array_equal(dataset.feature_mean, self.train.feature_mean.astype(np.float32))
            np.testing.assert_array_equal(dataset.aux_target_mean, self.train.aux_target_mean.astype(np.float32))

    def test_one_bar_target_uses_entry_close(self):
        dataset = TradingDatasetV3(self.path, **{**self.config, "prediction_horizon": 1}, return_aux_targets=True)
        entry = dataset.seq_len + len(dataset) - 1
        auxiliary = dataset[len(dataset) - 1][2].numpy() * dataset.aux_target_std + dataset.aux_target_mean
        self.assertEqual(entry, len(dataset.spot_close) - 1)
        expected = dataset.spot_close.iloc[entry] / dataset.spot_open.iloc[entry] - 1
        self.assertAlmostEqual(float(auxiliary[0]), expected, places=6)

    def test_future_changes_do_not_change_training_samples(self):
        changed = self.frame.copy()
        boundary = pd.Timestamp(self.train.split_boundaries["train_end"])
        changed.loc[changed.timestamp >= boundary, "spot_cvd"] += 10000
        path = Path(self.directory.name) / "changed.csv"
        changed.to_csv(path, index=False)
        dataset = TradingDatasetV3(path, **self.config)
        np.testing.assert_array_equal(dataset.features, self.train.features)
        np.testing.assert_array_equal(dataset.patterns[8:8 + len(dataset)], self.train.patterns[8:8 + len(dataset)])
        np.testing.assert_array_equal(dataset.aux_target_mean, self.train.aux_target_mean)

    def test_entry_bar_does_not_supply_pattern_conditions(self):
        detector = self.train
        frame = self.frame.copy()
        frame["open_interest"] = frame["open_interest_value"] = frame["oi_available"] = 0.0
        for transform in (detector._add_volume_profile, detector._add_basis_features,
                          detector._add_open_interest_features, detector._add_market_microstructure):
            frame = transform(frame)
        entry = 350
        original = detector._detect_patterns(frame.copy())
        changed = frame.copy()
        changed.loc[entry, ["spot_cvd", "futures_cvd", "basis_zscore", "spot_volume"]] = 1e9
        altered = detector._detect_patterns(changed)
        pattern_columns = [name for name in original if name.startswith("pattern_")]
        np.testing.assert_array_equal(original.loc[entry, pattern_columns], altered.loc[entry, pattern_columns])

    def test_evaluation_rejects_missing_statistics(self):
        with self.assertRaisesRegex(ValueError, "statistics from train"):
            TradingDatasetV3(self.path, **self.config, split="test")

    def test_gaps_and_duplicates_are_rejected(self):
        for name, frame in (("gap", self.frame.drop(200)),
                            ("duplicate", pd.concat([self.frame, self.frame.iloc[[200]]]))):
            with self.subTest(name=name):
                path = Path(self.directory.name) / f"{name}.csv"
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    TradingDatasetV3(path, **self.config)

    def test_short_split_and_invalid_statistics(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            TradingDatasetV3(self.path, **{**self.config, "seq_len": 1000})
        with self.assertRaises(ValueError):
            TradingDatasetV3(self.path, **self.config, normalization_stats={
                "feature_mean": np.zeros(28), "feature_std": -np.ones(28),
            })
