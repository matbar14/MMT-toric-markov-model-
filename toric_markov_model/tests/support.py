"""Deterministic synthetic market data, not a performance benchmark."""

import numpy as np
import pandas as pd


def market_frame(rows=720):
    random = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(random.normal(0, 0.008, rows)))
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")})
    for market, prices in (("spot", close), ("futures", close * (1 + random.normal(0, 0.002, rows)))):
        frame[f"{market}_open"] = prices * 0.999
        frame[f"{market}_high"] = prices * 1.01
        frame[f"{market}_low"] = prices * 0.99
        frame[f"{market}_close"] = prices
        frame[f"{market}_volume"] = random.uniform(100, 200, rows)
        frame[f"{market}_cvd"] = random.normal(0, 20, rows).cumsum() + 1000
    frame["basis"] = frame.futures_close - frame.spot_close
    frame["basis_pct"] = frame.basis / frame.spot_close * 100
    return frame
