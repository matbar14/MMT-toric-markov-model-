"""Enhanced trading dataset with Volume Profile and price levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TradingDatasetV2(Dataset):
    """Enhanced dataset with Volume Profile (POC, VAH, VAL) and price levels."""
    
    def __init__(
        self,
        csv_path: str,
        seq_len: int = 64,
        prediction_horizon: int = 1,
        train: bool = True,
        train_split: float = 0.8,
        volume_profile_period: int = 20,
    ):
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.volume_profile_period = volume_profile_period
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Calculate all indicators
        df = self._add_volume_profile(df)
        df = self._add_support_resistance(df)
        df = self._add_technical_indicators(df)
        
        # Remove NaN rows
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
    
    def _add_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Profile indicators: POC, VAH, VAL."""
        poc_list = []
        vah_list = []
        val_list = []
        
        for i in range(len(df)):
            if i < self.volume_profile_period:
                # Not enough data yet
                poc_list.append(df['close'].iloc[i])
                vah_list.append(df['high'].iloc[i])
                val_list.append(df['low'].iloc[i])
                continue
            
            # Get period data
            period_df = df.iloc[i - self.volume_profile_period:i]
            
            # Create price bins
            price_min = period_df['low'].min()
            price_max = period_df['high'].max()
            num_bins = 50
            bins = np.linspace(price_min, price_max, num_bins + 1)
            
            # Calculate volume at each price level
            volume_at_price = np.zeros(num_bins)
            
            for _, row in period_df.iterrows():
                # Distribute volume across price range of candle
                low_idx = np.digitize(row['low'], bins) - 1
                high_idx = np.digitize(row['high'], bins) - 1
                low_idx = max(0, min(low_idx, num_bins - 1))
                high_idx = max(0, min(high_idx, num_bins - 1))
                
                # Distribute volume evenly across the range
                if high_idx >= low_idx:
                    volume_per_bin = row['volume'] / (high_idx - low_idx + 1)
                    volume_at_price[low_idx:high_idx + 1] += volume_per_bin
            
            # POC: price with maximum volume
            poc_idx = np.argmax(volume_at_price)
            poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
            
            # Value Area: 70% of volume
            total_volume = volume_at_price.sum()
            target_volume = total_volume * 0.70
            
            # Find value area by expanding from POC
            cumsum_volume = volume_at_price[poc_idx]
            val_idx = poc_idx
            vah_idx = poc_idx
            
            while cumsum_volume < target_volume and (val_idx > 0 or vah_idx < num_bins - 1):
                # Expand to side with more volume
                vol_below = volume_at_price[val_idx - 1] if val_idx > 0 else 0
                vol_above = volume_at_price[vah_idx + 1] if vah_idx < num_bins - 1 else 0
                
                if vol_above >= vol_below and vah_idx < num_bins - 1:
                    vah_idx += 1
                    cumsum_volume += volume_at_price[vah_idx]
                elif val_idx > 0:
                    val_idx -= 1
                    cumsum_volume += volume_at_price[val_idx]
                else:
                    break
            
            vah = (bins[vah_idx] + bins[vah_idx + 1]) / 2
            val = (bins[val_idx] + bins[val_idx + 1]) / 2
            
            poc_list.append(poc)
            vah_list.append(vah)
            val_list.append(val)
        
        df['poc'] = poc_list
        df['vah'] = vah_list
        df['val'] = val_list
        
        # Distance to key levels (normalized)
        df['distance_to_poc'] = (df['close'] - df['poc']) / df['close']
        df['distance_to_vah'] = (df['close'] - df['vah']) / df['close']
        df['distance_to_val'] = (df['close'] - df['val']) / df['close']
        
        # Position in value area
        df['value_area_position'] = (df['close'] - df['val']) / (df['vah'] - df['val'] + 1e-8)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support/resistance levels."""
        # Pivot Points (классические)
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        # Distance to pivot levels
        df['distance_to_pivot'] = (df['close'] - df['pivot']) / df['close']
        df['distance_to_r1'] = (df['close'] - df['r1']) / df['close']
        df['distance_to_s1'] = (df['close'] - df['s1']) / df['close']
        
        # Previous period high/low
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['distance_to_prev_high'] = (df['close'] - df['prev_high']) / df['close']
        df['distance_to_prev_low'] = (df['close'] - df['prev_low']) / df['close']
        
        # Session high/low (rolling 24 periods for hourly data)
        df['session_high'] = df['high'].rolling(window=24).max()
        df['session_low'] = df['low'].rolling(window=24).min()
        df['distance_to_session_high'] = (df['close'] - df['session_high']) / df['close']
        df['distance_to_session_low'] = (df['close'] - df['session_low']) / df['close']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators including divergences."""
        # Price change for other calculations
        df['price_change'] = df['close'].diff()
        
        # Check if CVD already exists in data (from 5m aggregation)
        if 'cvd' not in df.columns:
            # Calculate CVD from hourly data (less accurate)
            df['volume_delta'] = np.where(df['price_change'] > 0, df['volume'], 
                                          np.where(df['price_change'] < 0, -df['volume'], 0))
            df['cvd'] = df['volume_delta'].cumsum()
        
        # Volume indicators
        df['obv'] = (np.sign(df['price_change']) * df['volume']).fillna(0).cumsum()
        df['cvd_normalized'] = df['cvd'] / df['cvd'].rolling(window=100).std().fillna(1)
        
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']
        
        df['buy_volume'] = np.where(df['price_change'] > 0, df['volume'], 0)
        df['sell_volume'] = np.where(df['price_change'] < 0, df['volume'], 0)
        df['buy_sell_ratio'] = (df['buy_volume'].rolling(20).sum() / 
                                (df['sell_volume'].rolling(20).sum() + 1e-8))
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['price_to_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['atr_normalized'] = df['atr'] / df['close']
        
        # ADX for market regime detection
        price_range = df['high'] - df['low']
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / (df['atr'] + 1e-8)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / (df['atr'] + 1e-8)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(14).mean()
        
        # Volume spike for breakout detection
        volume_ma = df['volume'].rolling(20).mean()
        df['volume_spike'] = df['volume'] / (volume_ma + 1e-8)
        
        # DIVERGENCES - CRITICAL FOR TRADING!
        # Правильный расчет дивергенций: сравниваем направления изменений
        lookback = 14
        
        # Изменения цены за период
        price_change = df['close'].diff(lookback)
        
        # Изменения индикаторов за тот же период
        cvd_change = df['cvd'].diff(lookback)
        obv_change = df['obv'].diff(lookback)
        poc_change = df['poc'].diff(lookback)
        
        # Дивергенция = когда направления расходятся
        # Бычья дивергенция: цена падает (price_change < 0), но индикатор растет (cvd_change > 0)
        # Медвежья дивергенция: цена растет (price_change > 0), но индикатор падает (cvd_change < 0)
        
        # CVD дивергенция
        cvd_bullish = ((price_change < -0.01) & (cvd_change > 0)).astype(float)
        cvd_bearish = ((price_change > 0.01) & (cvd_change < 0)).astype(float)
        df['cvd_price_divergence'] = cvd_bullish - cvd_bearish  # -1: медвежья, 0: нет, +1: бычья
        
        # OBV дивергенция
        obv_bullish = ((price_change < -0.01) & (obv_change > 0)).astype(float)
        obv_bearish = ((price_change > 0.01) & (obv_change < 0)).astype(float)
        df['obv_price_divergence'] = obv_bullish - obv_bearish
        
        # POC дивергенция (POC не следует за ценой)
        poc_bullish = ((price_change < -0.01) & (poc_change > 0)).astype(float)
        poc_bearish = ((price_change > 0.01) & (poc_change < 0)).astype(float)
        df['poc_price_divergence'] = poc_bullish - poc_bearish
        
        # Сила дивергенции: учитываем объем
        volume_spike = df['volume_spike'].fillna(1)
        df['cvd_div_strength'] = np.abs(df['cvd_price_divergence']) * volume_spike
        df['obv_div_strength'] = np.abs(df['obv_price_divergence']) * volume_spike
        
        # Momentum для других целей
        df['price_momentum'] = df['close'].pct_change(lookback)
        df['cvd_momentum'] = df['cvd'].pct_change(lookback)
        df['obv_momentum'] = df['obv'].pct_change(lookback)
        df['poc_momentum'] = df['poc'].pct_change(lookback)
        
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features with PROPER normalization for trading."""
        
        # CRITICAL: Use rolling statistics for normalization
        window = 100
        
        # Price features - RELATIVE to recent history
        df['price_zscore'] = ((df['close'] - df['close'].rolling(window).mean()) / 
                              (df['close'].rolling(window).std() + 1e-8)).fillna(0)
        df['high_zscore'] = ((df['high'] - df['high'].rolling(window).mean()) / 
                             (df['high'].rolling(window).std() + 1e-8)).fillna(0)
        df['low_zscore'] = ((df['low'] - df['low'].rolling(window).mean()) / 
                            (df['low'].rolling(window).std() + 1e-8)).fillna(0)
        
        # Volume features - RELATIVE
        df['volume_zscore'] = ((df['volume'] - df['volume'].rolling(window).mean()) / 
                               (df['volume'].rolling(window).std() + 1e-8)).fillna(0)
        
        # CVD - RELATIVE
        df['cvd_zscore'] = ((df['cvd'] - df['cvd'].rolling(window).mean()) / 
                            (df['cvd'].rolling(window).std() + 1e-8)).fillna(0)
        df['cvd_trend'] = (df['cvd'].diff(20) / 
                          (df['cvd'].rolling(window).std() + 1e-8)).fillna(0)
        
        # OBV - RELATIVE
        df['obv_zscore'] = ((df['obv'] - df['obv'].rolling(window).mean()) / 
                            (df['obv'].rolling(window).std() + 1e-8)).fillna(0)
        
        feature_cols = [
            # RELATIVE PRICE
            'price_zscore', 'high_zscore', 'low_zscore',
            'returns',
            
            # RELATIVE VOLUME
            'volume_zscore',
            
            # VOLUME PROFILE
            'distance_to_poc', 'distance_to_vah', 'distance_to_val',
            'value_area_position',
            
            # SUPPORT/RESISTANCE
            'distance_to_pivot',
            'distance_to_r1', 'distance_to_s1',
            'distance_to_prev_high', 'distance_to_prev_low',
            'distance_to_session_high', 'distance_to_session_low',
            
            # RELATIVE VOLUME INDICATORS
            'cvd_zscore', 'cvd_trend',
            'obv_zscore',
            'cvd_normalized',
            'vwap_distance',
            'buy_sell_ratio',
            
            # TECHNICAL
            'price_to_sma20',
            'rsi_normalized',
            'volatility',
            'atr_normalized',
            
            # DIVERGENCES
            'cvd_price_divergence',
            'obv_price_divergence',
            'poc_price_divergence',
            'cvd_div_strength',
            'obv_div_strength',
            
            # MOMENTUM
            'price_momentum',
            'cvd_momentum',
            'obv_momentum',
            'poc_momentum',
            
            # MARKET REGIME
            'adx',
            'volume_spike',
        ]
        
        # Extract and fill any remaining NaN
        features = df[feature_cols].fillna(0).values
        
        # Check for inf
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
        
        return features
    
    def _extract_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Extract all targets for multi-task learning.
        
        CRITICAL: Predict DEVIATION from session average, not absolute direction!
        Trading logic: compare future price to current session's average price.
        """
        # Calculate session average price (typical price)
        # This is the "fair value" for the current period
        session_avg_price = (df['high'] + df['low'] + df['close']) / 3
        
        # For better accuracy, use VWAP if we have volume
        # VWAP = cumulative(price * volume) / cumulative(volume) over session
        # Simplified: use rolling VWAP over last 24 hours (24 candles for hourly data)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap_24h = (typical_price * df['volume']).rolling(24).sum() / df['volume'].rolling(24).sum()
        
        # Use VWAP as "expected price" - this is what market considers fair
        expected_price = vwap_24h.fillna(session_avg_price)
        
        # Calculate future price
        future_price = df['close'].shift(-self.prediction_horizon)
        
        # DEVIATION: How much will future price deviate from current expected price?
        # Normalize by current price to get percentage deviation
        price_deviation = (future_price - expected_price) / expected_price
        
        # Classify based on DEVIATION from expected price
        # Use rolling std of price_deviation to set dynamic thresholds
        std_dev = price_deviation.rolling(window=100, min_periods=1).std()
        
        direction = np.zeros(len(df))
        # Significant positive deviation (price will be above fair value)
        direction[price_deviation > 0.5 * std_dev] = 2  # Up
        # Significant negative deviation (price will be below fair value)
        direction[price_deviation < -0.5 * std_dev] = 0  # Down
        # Within fair value range
        direction[(price_deviation >= -0.5 * std_dev) & (price_deviation <= 0.5 * std_dev)] = 1  # Neutral
        
        # Auxiliary task 1: Volume DEVIATION from session average
        session_avg_volume = df['volume'].rolling(24).mean()
        future_volume = df['volume'].shift(-self.prediction_horizon)
        volume_deviation = (future_volume - session_avg_volume) / (session_avg_volume + 1e-8)
        
        # Auxiliary task 2: CVD DEVIATION from session average
        session_avg_cvd = df['cvd'].rolling(24).mean()
        future_cvd = df['cvd'].shift(-self.prediction_horizon)
        cvd_deviation = (future_cvd - session_avg_cvd) / (df['cvd'].rolling(20).std() + 1e-8)
        
        # Auxiliary task 3: POC DEVIATION from current price
        # POC shows where most volume traded - deviation shows if price moves away from POC
        future_poc = df['poc'].shift(-self.prediction_horizon)
        poc_deviation = (future_poc - df['close']) / df['close']
        
        # Auxiliary task 4: Market regime (use ADX from df - already aligned after dropna)
        market_regime = np.zeros(len(df))
        adx_values = df['adx'].fillna(0).values
        market_regime[adx_values < 20] = 0  # Flat
        market_regime[(adx_values >= 20) & (adx_values < 40)] = 1  # Weak trend
        market_regime[adx_values >= 40] = 2  # Strong trend
        
        # Auxiliary task 5: Reversal detection
        price_momentum = df['close'].diff(5).fillna(0).values
        cvd_momentum = df['cvd_normalized'].diff(5).fillna(0).values
        volume_spike = df['volume_spike'].fillna(1).values
        
        reversal_signal = np.zeros(len(df))
        bullish_reversal = (price_momentum < -0.01) & (cvd_momentum > 0.5) & (volume_spike > 1.5)
        bearish_reversal = (price_momentum > 0.01) & (cvd_momentum < -0.5) & (volume_spike > 1.5)
        reversal_signal[bullish_reversal | bearish_reversal] = 1
        
        # Auxiliary task 6: Breakout detection
        range_high = df['high'].rolling(20).max()
        range_low = df['low'].rolling(20).min()
        
        breakout_up = ((df['close'] > range_high.shift(1)) & (volume_spike > 1.3)).fillna(False).values
        breakout_down = ((df['close'] < range_low.shift(1)) & (volume_spike > 1.3)).fillna(False).values
        
        breakout_signal = np.zeros(len(df))
        breakout_signal[breakout_up] = 1
        breakout_signal[breakout_down] = 2
        
        # Stack all targets - DEVIATIONS from session averages
        targets = np.stack([
            direction,  # Direction: future price vs VWAP (fair value)
            price_deviation.fillna(0).values,  # Price deviation from VWAP
            volume_deviation.fillna(0).values,  # Volume deviation from session avg
            cvd_deviation.fillna(0).values,  # CVD deviation from session avg
            poc_deviation.fillna(0).values,  # POC deviation from current price
            market_regime,
            reversal_signal,
            breakout_signal,
        ], axis=1)
        
        return targets
    
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.prediction_horizon
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx:idx + self.seq_len]
        target = self.targets[idx + self.seq_len]
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )
