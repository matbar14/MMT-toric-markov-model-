import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

script = Path(__file__).resolve().parents[1] / "scripts/download_market_year.py"
spec = importlib.util.spec_from_file_location("download_year", script)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def candle(hour, close_time=None):
    return [hour * downloader.HOUR_MS, "100", "102", "98", "101", "10",
            (hour + 1) * downloader.HOUR_MS - 1 if close_time is None else close_time,
            "1000", 10, "6", "600", "0"]


class Client:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.requests = []

    def get(self, *args, **kwargs):
        self.requests.append(kwargs)
        self.batch = next(self.batches)
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return self.batch

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class DownloadTests(unittest.TestCase):
    @patch.object(downloader.time, "sleep")
    def test_cli_resume_pins_window_and_rejects_completed_output(self, sleep):
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["download_market_year.py", "--days", "1", "--output-dir", directory]
            clocks = [{"serverTime": 24 * downloader.HOUR_MS}] * 2
            with patch.object(sys, "argv", arguments), patch.object(downloader, "session", return_value=Client(
                    clocks + [[candle(0)]])):
                with self.assertRaises(StopIteration):
                    downloader.main()
            clocks = [{"serverTime": 48 * downloader.HOUR_MS}] * 2
            client = Client(clocks + [[candle(hour) for hour in range(1, 24)],
                                     [candle(hour) for hour in range(24)]])
            with patch.object(sys, "argv", arguments + ["--resume"]), patch.object(
                    downloader, "session", return_value=client):
                downloader.main()
                with self.assertRaisesRegex(ValueError, "completed dataset"):
                    downloader.main()
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["end_exclusive"], "1970-01-02T00:00:00+00:00")
            self.assertEqual(manifest["rows"], 24)

    @patch.object(downloader.time, "sleep")
    def test_internal_gap_is_retried_without_synthetic_prices(self, sleep):
        client = Client([[candle(0), candle(2)], [candle(1)]])
        frame = downloader.download(client, "endpoint", "BTCUSDT", 0, 3 * downloader.HOUR_MS)
        self.assertEqual(len(frame), 3)
        self.assertEqual(client.requests[1]["params"]["startTime"], downloader.HOUR_MS)
        self.assertEqual(client.requests[1]["params"]["endTime"], 2 * downloader.HOUR_MS - 1)

    @patch.object(downloader.time, "sleep")
    def test_persistent_gap_is_reported_and_resume_fetches_only_missing_hour(self, sleep):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "spot_checkpoint.json"
            with self.assertRaisesRegex(ValueError, "1970-01-01T01:00:00\\+00:00"):
                downloader.download(Client([[candle(0), candle(2)], [], []]), "endpoint", "BTCUSDT",
                                    0, 3 * downloader.HOUR_MS, checkpoint)
            saved = json.loads(checkpoint.read_text())
            self.assertEqual(len(saved["rows"]), 2)
            report = json.loads(checkpoint.with_suffix(".gaps.json").read_text())
            self.assertEqual(report["missing_count"], 1)
            client = Client([[candle(1)]])
            frame = downloader.download(client, "endpoint", "BTCUSDT", 0, 3 * downloader.HOUR_MS, checkpoint)
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(len(frame), 3)
            self.assertEqual(json.loads(checkpoint.with_suffix(".gaps.json").read_text())["missing_count"], 0)

    @patch.object(downloader.time, "sleep")
    def test_network_failure_preserves_pages_and_identity_is_checked(self, sleep):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "spot_checkpoint.json"
            with self.assertRaises(StopIteration):
                downloader.download(Client([[candle(0)]]), "endpoint", "BTCUSDT", 0,
                                    2 * downloader.HOUR_MS, checkpoint)
            with self.assertRaisesRegex(ValueError, "different download"):
                downloader.download(Client([]), "other-endpoint", "BTCUSDT", 0,
                                    2 * downloader.HOUR_MS, checkpoint)
            client = Client([[candle(1)]])
            frame = downloader.download(client, "endpoint", "BTCUSDT", 0, 2 * downloader.HOUR_MS, checkpoint)
            self.assertEqual(len(frame), 2)
            self.assertEqual(client.requests[0]["params"]["startTime"], downloader.HOUR_MS)
            self.assertEqual(len(downloader.download(Client([]), "endpoint", "BTCUSDT", 0,
                                                    2 * downloader.HOUR_MS, checkpoint)), 2)

    @patch.object(downloader.time, "sleep")
    def test_invalid_prices_are_not_checkpointed(self, sleep):
        invalid = candle(1)
        invalid[2] = "1"
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "spot_checkpoint.json"
            with self.assertRaisesRegex(ValueError, "OHLC"):
                downloader.download(Client([[candle(0)], [invalid]]), "endpoint", "BTCUSDT", 0,
                                    2 * downloader.HOUR_MS, checkpoint)
            self.assertEqual(len(json.loads(checkpoint.read_text())["rows"]), 1)

    @patch.object(downloader.time, "sleep")
    def test_complete_download_and_cvd(self, sleep):
        frame = downloader.download(Client([[candle(0)], [candle(1)]]), "endpoint", "BTCUSDT", 0,
                                    2 * downloader.HOUR_MS)
        merged = downloader.merge(frame, frame)
        self.assertEqual(merged.spot_cvd.tolist(), [2, 4])
        self.assertEqual(merged.basis.tolist(), [0, 0])
        self.assertNotIn("open_interest", merged)

    @patch.object(downloader.time, "sleep")
    def test_gaps_empty_and_incomplete_are_rejected(self, sleep):
        for batch in ([], [candle(1)], [candle(0, 2 * downloader.HOUR_MS)], [candle(0), candle(0)]):
            with self.subTest(batch=batch):
                with self.assertRaises((ValueError, StopIteration)):
                    downloader.download(Client([batch]), "endpoint", "BTCUSDT", 0, 2 * downloader.HOUR_MS)
