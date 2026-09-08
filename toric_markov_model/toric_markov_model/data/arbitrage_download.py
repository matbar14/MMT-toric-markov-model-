"""Resumable public mark-price and settled-funding inputs for BTCUSDT research."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..arbitrage import ArbitrageData, validate_funding, validate_marks
from .payoff import validate_market


HOUR_MS = 3_600_000
ENDPOINTS = dict(funding="https://fapi.binance.com/fapi/v1/fundingRate",
                 marks="https://fapi.binance.com/fapi/v1/markPriceKlines")


def session():
    client = requests.Session()
    client.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))))
    return client


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalized(rows, kind, symbol):
    if kind == "funding":
        if any(not isinstance(row, dict) or row.get("symbol") != symbol for row in rows):
            raise ValueError("funding schema or symbol mismatch")
        frame = pd.DataFrame([dict(timestamp=pd.to_datetime(row["fundingTime"], unit="ms", utc=True),
                                   funding_rate=float(row["fundingRate"]), mark_price=float(row["markPrice"]))
                              for row in rows], columns=["timestamp", "funding_rate", "mark_price"])
        if len(frame) >= 2:
            validate_funding(frame)
        if len(frame) and (not np.isfinite(frame[["funding_rate", "mark_price"]].to_numpy()).all() or
                           (frame.mark_price <= 0).any() or (frame.funding_rate.abs() >= 1).any()):
            raise ValueError("invalid funding values")
        return frame
    if any(not isinstance(row, list) or len(row) != 12 or int(row[6]) != int(row[0]) + HOUR_MS - 1 for row in rows):
        raise ValueError("invalid mark kline schema or close time")
    frame = pd.DataFrame([dict(timestamp=pd.to_datetime(row[0], unit="ms", utc=True),
                               **dict(zip(("open", "high", "low", "close"), map(float, row[1:5]))))
                          for row in rows], columns=["timestamp", "open", "high", "low", "close"])
    if len(frame):
        expected = pd.Series(pd.date_range(frame.timestamp.iloc[0], periods=len(frame), freq="h"))
        validate_marks(frame, expected)
        if not frame.timestamp.equals(frame.timestamp.dt.floor("h")):
            raise ValueError("unaligned mark candles")
    return frame


def fetch_history(client, kind, symbol, start_ms, end_ms, checkpoint):
    identity = dict(kind=kind, endpoint=ENDPOINTS[kind], symbol=symbol, start_ms=start_ms, end_ms=end_ms)
    rows, cursor = [], start_ms
    step = 1 if kind == "funding" else HOUR_MS
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("request") != identity:
            raise ValueError("checkpoint belongs to a different market/window")
        rows = saved["rows"]
        normalized(rows, kind, symbol)
        stamps = [int(row["fundingTime"] if kind == "funding" else row[0]) for row in rows]
        if stamps and (stamps[0] < start_ms or stamps[-1] >= end_ms or any(
                right <= left for left, right in zip(stamps, stamps[1:]))):
            raise ValueError("invalid cached timestamps")
        if kind == "marks" and stamps and stamps[0] != start_ms:
            raise ValueError("mark cache has a missing initial hour")
        cursor = stamps[-1] + step if stamps else start_ms
    while cursor < end_ms:
        params = dict(symbol=symbol, startTime=cursor, endTime=end_ms - 1, limit=1000)
        if kind == "marks":
            params["interval"] = "1h"
        response = client.get(ENDPOINTS[kind], params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError("invalid exchange response")
        if not batch:
            break
        normalized(batch, kind, symbol)
        stamps = [int(row["fundingTime"] if kind == "funding" else row[0]) for row in batch]
        if (stamps[0] < cursor or stamps[-1] >= end_ms or
                any(right <= left for left, right in zip(stamps, stamps[1:])) or
                (kind == "marks" and stamps[0] != cursor)):
            raise ValueError(f"invalid {kind} chronology or missing mark hours at {cursor}")
        normalized(rows + batch, kind, symbol)
        rows.extend(batch)
        cursor = stamps[-1] + step
        write_json(checkpoint, dict(request=identity, rows=rows))
        print(f"{kind}: {len(rows)} validated records saved", flush=True)
        time.sleep(0.15)
    return normalized(rows, kind, symbol)


def download_inputs(client, market_path, output, resume=False):
    market_path, output = Path(market_path), Path(output)
    frame = validate_market(pd.read_csv(market_path))
    market_hash = hashlib.sha256(market_path.read_bytes()).hexdigest()
    source_manifest = json.loads(market_path.with_name("manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("symbol") != "BTCUSDT" or source_manifest.get("sha256") != market_hash:
        raise ValueError("verified BTCUSDT market.csv and original manifest.json required")
    symbol = "BTCUSDT"
    start_ms = int(frame.timestamp.iloc[0].timestamp() * 1000)
    end_ms = int((frame.timestamp.iloc[-1] + pd.Timedelta(hours=1)).timestamp() * 1000)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(symbol=symbol, market_sha256=market_hash, start_ms=start_ms, end_ms=end_ms)
    request_path = output / "request.json"
    if (output / "manifest.json").exists():
        raise ValueError("completed arbitrage inputs already exist; use a new directory")
    if any(output.iterdir()):
        if not resume or not request_path.exists() or json.loads(request_path.read_text()) != identity:
            raise ValueError("use a new output directory or resume the identical market window")
    write_json(request_path, identity)
    marks = fetch_history(client, "marks", symbol, start_ms, end_ms, output / "marks_checkpoint.json")
    funding = fetch_history(client, "funding", symbol, start_ms - 12 * HOUR_MS, end_ms + 12 * HOUR_MS,
                            output / "funding_checkpoint.json")
    ArbitrageData.from_frames(frame, funding, marks)
    hashes = {}
    for name, values in (("funding.csv", funding), ("marks.csv", marks)):
        path = output / name
        values.to_csv(path, index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(format="arbitrage_inputs_v1", **identity, files=hashes, endpoints=ENDPOINTS,
                    funding="settled events, no fixed eight-hour interpolation; twelve-hour fetch padding",
                    limits="hourly mark bars and trade candles are not synchronized executable quotes")
    write_json(output / "manifest.json", manifest)
    return manifest


def load_inputs(market_path, directory, config=None):
    directory, market_path = Path(directory), Path(market_path)
    if not (directory / "manifest.json").exists():
        raise ValueError("funding/mark dataset missing; run download_arbitrage_inputs.py first")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "arbitrage_inputs_v1" or manifest.get("symbol") != "BTCUSDT" or
            manifest.get("market_sha256") != hashlib.sha256(market_path.read_bytes()).hexdigest()):
        raise ValueError("arbitrage provenance does not match BTCUSDT market data")
    for name in ("funding.csv", "marks.csv"):
        if manifest["files"].get(name) != hashlib.sha256((directory / name).read_bytes()).hexdigest():
            raise ValueError(f"arbitrage checksum mismatch: {name}")
    return ArbitrageData.from_frames(pd.read_csv(market_path), pd.read_csv(directory / "funding.csv"),
                                     pd.read_csv(directory / "marks.csv"), config), manifest
