#!/usr/bin/env python3
"""Download public BTCUSDT settled funding and hourly mark prices; never submit orders."""

import argparse
import json
from pathlib import Path

from toric_markov_model.data.arbitrage_download import download_inputs, session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with session() as client:
        manifest = download_inputs(client, args.market_data, args.output_dir, args.resume)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
