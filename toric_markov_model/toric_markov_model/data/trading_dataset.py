"""Trading dataset with technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TradingDataset(Dataset):
    """Dataset for trading with OHLCV data and technical indicators."""
    
    def __init__(
        self,
        csv_path: str,
        seq_len: int = 64,
        prediction_horizon: int = 1,
        train: bool = True,
        train_split: float = 0.8,
    ):
        """
        Args:
            csv_path: Path to CSV with columns: timestamp, open, high, low, close, volume
            seq_len: Length of input sequence
            prediction_horizon: How many steps ahead to predict
            train: If True, use training split, else validation split
            train_split: Fraction of data for training
        """
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Calculate technical indicators
        df = self._add_technical_indicators(df)
        
        # Remove NaN rows (from indicators)
        df = df.dropna()
        
        # Split train/val
        split_idx = int(len(df) * train_split)
        if train:
            df = df.iloc[:split_idx]
        else:
            df = df.iloc[split_idx:]
        
        # Extract features and targets
        self.features = self._extract_features(df)
        self.targets = self._extract_targets(df)
        
        # Normalize features
        self.feature_mean = self.features.mean(axis=0)
        self.feature_std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        print(f"{'Train' if train else 'Val'} dataset: {len(self)} samples, "
              f"{self.features.shape[1]} features")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        # === VOLUME INDICATORS (MOST IMPORTANT) ===
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume Delta (approximation: up volume - down volume)
        df['price_change'] = df['close'].diff()
        df['volume_delta'] = np.where(df['price_change'] > 0, df['volume'], 
                                      np.where(df['price_change'] < 0, -df['volume'], 0))
        
        # Cumulative Volume Delta (CVD)
        df['cvd'] = df['volume_delta'].cumsum()
        df['cvd_normalized'] = df['cvd'] / df['cvd'].rolling(window=100).std().fillna(1)
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Buy/Sell Pressure (approximation)
        df['buy_volume'] = np.where(df['price_change'] > 0, df['volume'], 0)
        df['sell_volume'] = np.where(df['price_change'] < 0, df['volume'], 0)
        df['buy_sell_ratio'] = (df['buy_volume'].rolling(20).sum() / 
                                (df['sell_volume'].rolling(20).sum() + 1e-8))
        
        # Volume Momentum
        df['volume_momentum'] = df['volume'].pct_change(5)
        
        # Volume Acceleration
        df['volume_acceleration'] = df['volume_momentum'].diff()
        
        # === PRICE INDICATORS ===
        
        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Price distance from MAs
        df['price_to_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_to_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
        
        # === MOMENTUM INDICATORS ===
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_normalized'] = df['macd_hist'] / df['close']
        
        # === VOLATILITY INDICATORS ===
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_normalized'] = df['atr'] / df['close']
        
        # === PRICE ACTION ===
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # High-Low range
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Body to wick ratio (candle pattern)
        df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature columns."""
        feature_cols = [
            # OHLCV
            'open', 'high', 'low', 'close', 'volume',
            
            # VOLUME INDICATORS (PRIORITY)
            'obv', 'cvd', 'cvd_normalized',
            'vwap', 'vwap_distance',
            'buy_sell_ratio',
            'volume_momentum', 'volume_acceleration',
            
            # MOVING AVERAGES
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20',
            'price_to_sma20', 'price_to_ema20',
            
            # MOMENTUM
            'rsi', 'rsi_normalized',
            'macd', 'macd_signal', 'macd_hist', 'macd_hist_normalized',
            
            # VOLATILITY
            'bb_width', 'bb_position',
            'atr', 'atr_normalized',
            
            # PRICE ACTION
            'returns', 'log_returns', 'volatility',
            'hl_ratio', 'body_size', 'upper_wick', 'lower_wick',
        ]
        return df[feature_cols].values
    
    def _extract_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target (future price direction and return)."""
        # Future return
        future_return = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Direction: 0=down, 1=neutral, 2=up
        direction = np.zeros(len(df))
        direction[future_return > 0.001] = 2  # Up
        direction[future_return < -0.001] = 0  # Down
        direction[(future_return >= -0.001) & (future_return <= 0.001)] = 1  # Neutral
        
        # Stack direction and return
        targets = np.stack([direction, future_return.fillna(0).values], axis=1)
        return targets
    
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.prediction_horizon
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get sequence of features
        features = self.features[idx:idx + self.seq_len]
        
        # Get target at the end of sequence
        target = self.targets[idx + self.seq_len]
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def create_trading_dataloader(
    csv_path: str,
    seq_len: int = 64,
    batch_size: int = 32,
    prediction_horizon: int = 1,
    train: bool = True,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create dataloader for trading."""
    from torch.utils.data import DataLoader
    
    dataset = TradingDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        prediction_horizon=prediction_horizon,
        train=train,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
