#!/usr/bin/env python3
"""Download crypto data from BOTH spot and futures with basis calculation."""

import argparse
import pandas as pd
import requests
import time


def download_binance_spot_data(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Download OHLCV data from Binance Spot."""
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    
    while len(all_data) < limit:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(1000, limit - len(all_data))
        }
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error downloading spot: {e}", flush=True)
            break
            
        if not data:
            break
        
        all_data = data + all_data
        end_time = data[0][0] - 1
        print(f"Downloaded {len(all_data)}/{limit} spot candles...", flush=True)
        
        if len(data) < 1000:
            break
        
        time.sleep(0.1)
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
        df[col] = df[col].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base']]


def download_binance_futures_data(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Download OHLCV data from Binance Futures."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_data = []
    end_time = None
    
    while len(all_data) < limit:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(1000, limit - len(all_data))
        }
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error downloading futures: {e}", flush=True)
            break
            
        if not data:
            break
        
        all_data = data + all_data
        end_time = data[0][0] - 1
        print(f"Downloaded {len(all_data)}/{limit} futures candles...", flush=True)
        
        if len(data) < 1000:
            break
        
        time.sleep(0.1)
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
        df[col] = df[col].astype(float)
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base']]


def download_open_interest(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Download Open Interest history from Binance Futures.

    Binance endpoint `/futures/data/openInterestHist` has an exchange-side limit:
    only the latest ~30 days of history are available, regardless of requested limit.
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_data = []
    end_time = None
    
    while len(all_data) < limit:
        params = {
            'symbol': symbol,
            'period': interval,
            'limit': min(500, limit - len(all_data))
        }
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error downloading OI: {e}", flush=True)
            break
            
        if not data:
            break
        
        all_data = data + all_data
        if len(data) > 0:
            end_time = data[0]['timestamp'] - 1
        print(f"Downloaded {len(all_data)}/{limit} OI records...", flush=True)
        
        if len(data) < 500:
            break
        
        time.sleep(0.1)
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
    df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
    
    return df[['timestamp', 'sumOpenInterest', 'sumOpenInterestValue']]


def calculate_cvd_5m(df_5m: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """Calculate CVD on 5-minute data using taker buy volumes."""
    print(f"Calculating CVD on 5m {prefix}data...", flush=True)
    
    # Taker buy = market buy (bullish)
    # Taker sell = total - taker buy (bearish)
    df_5m[f'{prefix}taker_sell_base'] = df_5m[f'{prefix}volume'] - df_5m[f'{prefix}taker_buy_base']
    
    # CVD = cumulative(buy_volume - sell_volume)
    df_5m[f'{prefix}volume_delta'] = df_5m[f'{prefix}taker_buy_base'] - df_5m[f'{prefix}taker_sell_base']
    df_5m[f'{prefix}cvd'] = df_5m[f'{prefix}volume_delta'].cumsum()
    
    return df_5m


def merge_spot_futures(df_spot_5m: pd.DataFrame, df_futures_5m: pd.DataFrame, df_oi: pd.DataFrame) -> pd.DataFrame:
    """Merge spot and futures data, calculate basis."""
    print("Merging spot and futures...", flush=True)
    
    # Rename columns
    df_spot_5m = df_spot_5m.rename(columns={
        'open': 'spot_open', 'high': 'spot_high', 'low': 'spot_low',
        'close': 'spot_close', 'volume': 'spot_volume',
        'taker_buy_base': 'spot_taker_buy_base'
    })
    
    df_futures_5m = df_futures_5m.rename(columns={
        'open': 'futures_open', 'high': 'futures_high', 'low': 'futures_low',
        'close': 'futures_close', 'volume': 'futures_volume',
        'taker_buy_base': 'futures_taker_buy_base'
    })
    
    # Merge on timestamp
    df_merged = df_spot_5m.merge(df_futures_5m, on='timestamp', how='inner')
    
    # Calculate CVD for both
    df_merged = calculate_cvd_5m(df_merged, 'spot_')
    df_merged = calculate_cvd_5m(df_merged, 'futures_')
    
    # Calculate BASIS (futures - spot)
    df_merged['basis'] = df_merged['futures_close'] - df_merged['spot_close']
    df_merged['basis_pct'] = (df_merged['basis'] / df_merged['spot_close']) * 100
    
    # Aggregate to 1h
    df_merged['hour'] = df_merged['timestamp'].dt.floor('1H')
    
    df_1h = df_merged.groupby('hour').agg({
        # Spot
        'spot_open': 'first', 'spot_high': 'max', 'spot_low': 'min',
        'spot_close': 'last', 'spot_volume': 'sum', 'spot_cvd': 'last',
        # Futures
        'futures_open': 'first', 'futures_high': 'max', 'futures_low': 'min',
        'futures_close': 'last', 'futures_volume': 'sum', 'futures_cvd': 'last',
        # Basis
        'basis': 'mean', 'basis_pct': 'mean'
    }).reset_index()
    
    df_1h.rename(columns={'hour': 'timestamp'}, inplace=True)
    
    # Merge Open Interest
    if not df_oi.empty:
        df_oi['hour'] = df_oi['timestamp'].dt.floor('1H')
        df_oi_agg = df_oi.groupby('hour').agg({
            'sumOpenInterest': 'last',
            'sumOpenInterestValue': 'last'
        }).reset_index()
        df_oi_agg.rename(columns={'hour': 'timestamp'}, inplace=True)
        
        df_1h = df_1h.merge(df_oi_agg, on='timestamp', how='left')
        df_1h['sumOpenInterest'] = df_1h['sumOpenInterest'].ffill()
        df_1h['sumOpenInterestValue'] = df_1h['sumOpenInterestValue'].ffill()
    else:
        df_1h['sumOpenInterest'] = 0
        df_1h['sumOpenInterestValue'] = 0
    
    df_1h.rename(columns={
        'sumOpenInterest': 'open_interest',
        'sumOpenInterestValue': 'open_interest_value'
    }, inplace=True)
    
    return df_1h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--output", type=str, default="btc_data_with_basis.csv")
    parser.add_argument("--hours", type=int, default=180)
    args = parser.parse_args()
    
    print(f"Downloading {args.symbol} from SPOT and FUTURES...", flush=True)
    print("=" * 80, flush=True)
    
    candles_5m = args.hours * 12
    
    print(f"Step 1: Downloading SPOT 5m candles...", flush=True)
    df_spot_5m = download_binance_spot_data(args.symbol, '5m', candles_5m)
    print(f"✓ Downloaded {len(df_spot_5m)} spot candles", flush=True)
    
    print(f"\nStep 2: Downloading FUTURES 5m candles...", flush=True)
    df_futures_5m = download_binance_futures_data(args.symbol, '5m', candles_5m)
    print(f"✓ Downloaded {len(df_futures_5m)} futures candles", flush=True)
    
    print(f"\nStep 3: Downloading Open Interest...", flush=True)
    df_oi = download_open_interest(args.symbol, '1h', args.hours)
    if not df_oi.empty:
        print(f"✓ Downloaded {len(df_oi)} OI records", flush=True)
        if len(df_oi) < args.hours:
            print(
                "⚠ OI history is shorter than requested hours. "
                "Binance openInterestHist provides only the latest ~30 days.",
                flush=True,
            )
    else:
        print("⚠ No OI data available", flush=True)
    
    print(f"\nStep 4: Merging and calculating basis...", flush=True)
    df_1h = merge_spot_futures(df_spot_5m, df_futures_5m, df_oi)
    print(f"✓ {len(df_1h)} 1h candles", flush=True)
    
    print(f"\nStep 5: Saving...", flush=True)
    df_1h.to_csv(args.output, index=False)
    print(f"✓ Saved to {args.output}", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print(f"Candles: {len(df_1h)}", flush=True)
    print(f"Period: {df_1h['timestamp'].min()} to {df_1h['timestamp'].max()}", flush=True)
    print(f"\nSpot CVD range: {df_1h['spot_cvd'].min():.2f} to {df_1h['spot_cvd'].max():.2f}", flush=True)
    print(f"Futures CVD range: {df_1h['futures_cvd'].min():.2f} to {df_1h['futures_cvd'].max():.2f}", flush=True)
    print(f"Basis range: {df_1h['basis'].min():.2f} to {df_1h['basis'].max():.2f}", flush=True)
    print(f"Basis %: {df_1h['basis_pct'].min():.4f}% to {df_1h['basis_pct'].max():.4f}%", flush=True)
    if not df_oi.empty:
        print(f"OI range: {df_1h['open_interest'].min():.2f} to {df_1h['open_interest'].max():.2f}", flush=True)
        oi_cov = (df_1h['open_interest'].notna() & df_1h['open_interest_value'].notna()).mean() * 100
        print(f"OI coverage in merged 1h dataset: {oi_cov:.2f}% of rows", flush=True)
        if oi_cov < 20:
            print(
                "⚠ Low OI coverage: train/val split may place almost all OI into validation only. "
                "Consider training a no-OI baseline and a separate OI-only model.",
                flush=True,
            )
    
    print("\nFirst rows:")
    print(df_1h.head(), flush=True)
    print("\nColumns:", df_1h.columns.tolist(), flush=True)


if __name__ == "__main__":
    main()
