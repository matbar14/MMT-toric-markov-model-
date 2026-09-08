#!/usr/bin/env python3
"""Download a complete, closed hourly spot/futures window with reproducible provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HOUR_MS = 3_600_000
ENDPOINTS = {
    "spot": "https://api.binance.com/api/v3/klines",
    "futures": "https://fapi.binance.com/fapi/v1/klines",
}
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def session():
    client = requests.Session()
    retries = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=("GET",), respect_retry_after_header=True)
    client.mount("https://", HTTPAdapter(max_retries=retries))
    return client


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_rows(rows, start_ms, end_ms):
    if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != 12 for row in rows):
        raise ValueError("unexpected kline schema")
    stamps = [int(row[0]) for row in rows]
    if any(stamp != row[0] or stamp % HOUR_MS or not start_ms <= stamp < end_ms
           for stamp, row in zip(stamps, rows)) or any(right <= left for left, right in zip(stamps, stamps[1:])):
        raise ValueError("candles must be unique, chronological and inside the requested hourly window")
    if any(int(row[6]) != stamp + HOUR_MS - 1 for stamp, row in zip(stamps, rows)):
        raise ValueError("incomplete candle or invalid close time")
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for name in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    values = frame[["open", "high", "low", "close", "volume", "taker_buy_base"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite exchange values")
    if (values[["open", "high", "low", "close", "volume"]] <= 0).any().any():
        raise ValueError("nonpositive prices or volumes")
    if ((frame.taker_buy_base < 0) | (frame.taker_buy_base > frame.volume)).any():
        raise ValueError("invalid taker volume")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
            (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("invalid OHLC range")
    return frame


def download(client, endpoint, symbol, start_ms, end_ms, checkpoint=None, gap_retries=2):
    if start_ms % HOUR_MS or end_ms % HOUR_MS or start_ms >= end_ms or gap_retries < 0:
        raise ValueError("invalid hourly window or retry count")
    identity = dict(endpoint=endpoint, symbol=symbol, start_ms=start_ms, end_ms=end_ms, interval="1h")
    rows, cursor = [], start_ms
    if checkpoint is not None and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("version") != 1 or saved.get("request") != identity:
            raise ValueError("checkpoint belongs to a different download window or market")
        rows, cursor = saved["rows"], saved["cursor"]
        validate_rows(rows, start_ms, end_ms)
        if (not isinstance(cursor, int) or cursor % HOUR_MS or not start_ms <= cursor <= end_ms or
                (rows and cursor <= int(rows[-1][0]))):
            raise ValueError("invalid checkpoint cursor")

    def save():
        if checkpoint is not None:
            write_json(checkpoint, dict(version=1, request=identity, cursor=cursor, rows=rows))

    def fetch(first, last):
        response = client.get(endpoint, params={"symbol": symbol, "interval": "1h", "limit": 1000,
                                                "startTime": first, "endTime": last - 1}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        validate_rows(batch, first, last)
        time.sleep(0.15)
        return batch

    while cursor < end_ms:
        batch = fetch(cursor, end_ms)
        rows.extend(batch)
        cursor = int(batch[-1][0]) + HOUR_MS if batch else end_ms
        save()
        print(f"{symbol}: {len(rows)}/{(end_ms - start_ms) // HOUR_MS} closed candles", flush=True)
    expected = np.arange(start_ms, end_ms, HOUR_MS, dtype=np.int64)
    for attempt in range(gap_retries + 1):
        missing = np.setdiff1d(expected, [int(row[0]) for row in rows])
        report = dict(request=identity, missing_count=len(missing),
                      missing_utc=[pd.Timestamp(int(stamp), unit="ms", tz="UTC").isoformat() for stamp in missing])
        if checkpoint is not None:
            write_json(checkpoint.with_suffix(".gaps.json"), report)
        if not len(missing):
            break
        if attempt == gap_retries:
            raise ValueError(f"missing {len(missing)} hourly candles: {', '.join(report['missing_utc'][:24])}; "
                             "no synthetic candles created; progress retained when checkpoint is enabled")
        for group in np.split(missing, np.flatnonzero(np.diff(missing) != HOUR_MS) + 1):
            for offset in range(0, len(group), 1000):
                chunk = group[offset:offset + 1000]
                rows.extend(fetch(int(chunk[0]), int(chunk[-1]) + HOUR_MS))
                rows.sort(key=lambda row: int(row[0]))
                save()
    frame = validate_rows(rows, start_ms, end_ms)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame


def merge(spot, futures):
    if not spot.timestamp.equals(futures.timestamp):
        raise ValueError("spot/futures timestamps differ")
    result = pd.DataFrame({"timestamp": spot.timestamp})
    for market, frame in (("spot", spot), ("futures", futures)):
        for name in ("open", "high", "low", "close", "volume", "taker_buy_base"):
            result[f"{market}_{name}"] = frame[name]
        result[f"{market}_cvd"] = (2 * frame.taker_buy_base - frame.volume).cumsum()
    result["basis"] = result.futures_close - result.spot_close
    result["basis_pct"] = 100 * result.basis / result.spot_close
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--end", help="Exclusive UTC hour boundary; default last closed hour from exchange time")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume validated page checkpoints for the same window")
    args = parser.parse_args()
    if args.days < 1 or not args.symbol.isalnum():
        raise ValueError("positive days and alphanumeric symbol required")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "download_request.json"
    saved_request = None
    if args.resume and request_path.exists():
        saved_request = json.loads(request_path.read_text(encoding="utf-8"))
        if saved_request.get("symbol") != args.symbol.upper() or saved_request.get("days") != args.days:
            raise ValueError("resume symbol/days differ from saved request")
    elif any(output.iterdir()):
        raise ValueError("use a new empty output directory to avoid replacing an existing dataset")
    if (output / "market.csv").exists() or (output / "manifest.json").exists():
        raise ValueError("completed dataset already exists; use a new output directory")
    with session() as client:
        clocks = []
        for endpoint in ("https://api.binance.com/api/v3/time", "https://fapi.binance.com/fapi/v1/time"):
            response = client.get(endpoint, timeout=30)
            response.raise_for_status()
            clocks.append(int(response.json()["serverTime"]))
        if abs(clocks[0] - clocks[1]) > 60_000:
            raise ValueError("spot and futures server clocks disagree by more than one minute")
        latest_boundary = min(clocks) // HOUR_MS * HOUR_MS
        if args.end:
            end = pd.Timestamp(args.end)
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            end_ms = int(end.timestamp() * 1000)
        elif saved_request is not None:
            end_ms = saved_request["end_ms"]
        else:
            end_ms = latest_boundary
        if end_ms % HOUR_MS or end_ms > latest_boundary:
            raise ValueError("end must be a completed UTC hour boundary")
        start_ms = end_ms - args.days * 24 * HOUR_MS
        request = dict(symbol=args.symbol.upper(), days=args.days, start_ms=start_ms, end_ms=end_ms)
        if saved_request is not None and saved_request != request:
            raise ValueError("resume window differs from saved request")
        write_json(request_path, request)
        print("Window:", pd.Timestamp(start_ms, unit="ms", tz="UTC"), "through",
              pd.Timestamp(end_ms, unit="ms", tz="UTC"), "(exclusive)", flush=True)
        frames = {}
        for market, endpoint in ENDPOINTS.items():
            print(f"Downloading {market}", flush=True)
            frames[market] = download(client, endpoint, args.symbol.upper(), start_ms, end_ms,
                                      checkpoint=output / f"{market}_checkpoint.json")
            frames[market].to_csv(output / f"{market}_raw.csv", index=False)
    frame = merge(frames["spot"], frames["futures"])
    path = output / "market.csv"
    frame.to_csv(path, index=False)
    manifest = dict(
        symbol=args.symbol.upper(), interval="1h", days=args.days, rows=len(frame),
        start_inclusive=pd.Timestamp(start_ms, unit="ms", tz="UTC").isoformat(),
        end_exclusive=pd.Timestamp(end_ms, unit="ms", tz="UTC").isoformat(),
        exchange_server_time_ms=clocks, endpoints=ENDPOINTS,
        cvd="cumsum(2 * hourly taker_buy_base - hourly volume), reset at window start",
        basis="hourly futures close minus spot close; not mean 5m basis",
        open_interest="omitted consistently over the entire history; no synthetic OI",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
