"""Pattern-based trading dataset - detect specific tradeable patterns.

Instead of predicting direction on every candle, we detect PATTERNS:
- Bullish/Bearish divergences (price vs CVD/OI)
- Basis anomalies (futures-spot spread extremes)
- Volume profile breakouts (POC breaks with volume)
- Accumulation/Distribution (OI changes + price action)
- Liquidity grabs (stop hunts before reversals)

Model learns to recognize these patterns and predict if they will be profitable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PatternTradingDataset(Dataset):
    """Dataset that detects and labels trading patterns."""
    
    def __init__(
        self,
        csv_path: str,
        seq_len: int = 64,
        prediction_horizon: int = 4,  # 4 hours ahead
        train: bool = True,
        train_split: float = 0.8,
        volume_profile_period: int = 20,
        min_pattern_profit: float = 0.015,  # 1.5% minimum profit to label as pattern
    ):
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.volume_profile_period = volume_profile_period
        self.min_pattern_profit = min_pattern_profit
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Fill NaN in OI
        if 'open_interest' in df.columns:
            df['open_interest'] = df['open_interest'].ffill().bfill()
            df['open_interest_value'] = df['open_interest_value'].ffill().bfill()
        
        # Calculate all indicators
        df = self._add_volume_profile(df)
        df = self._add_basis_features(df)
        df = self._add_open_interest_features(df)
        df = self._add_market_microstructure(df)
        
        # CRITICAL: Detect patterns
        df = self._detect_patterns(df)
        
        # Remove NaN rows
        df = df.dropna()
        
        # Split train/val
        split_idx = int(len(df) * train_split)
        if train:
            df = df.iloc[:split_idx]
        else:
            df = df.iloc[split_idx:]
        
        # Extract features and pattern labels
        self.features = self._extract_features(df)
        self.patterns = self._extract_pattern_labels(df)
        
        # Normalize features
        self.feature_mean = self.features.mean(axis=0)
        self.feature_std = self.features.std(axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        # Count patterns
        pattern_counts = self.patterns.sum(axis=0)
        print(f"{'Train' if train else 'Val'} dataset: {len(self)} samples, "
              f"{self.features.shape[1]} features")
        print(f"  CVD Patterns:")
        print(f"    Bullish divergence: {int(pattern_counts[0])}")
        print(f"    Bearish divergence: {int(pattern_counts[1])}")
        print(f"    CVD reversal bull: {int(pattern_counts[2])}")
        print(f"    CVD reversal bear: {int(pattern_counts[3])}")
        print(f"    CVD exhaustion bull: {int(pattern_counts[4])}")
        print(f"    CVD exhaustion bear: {int(pattern_counts[5])}")
        print(f"    CVD spot-futures bull: {int(pattern_counts[6])}")
        print(f"    CVD spot-futures bear: {int(pattern_counts[7])}")
        print(f"    CVD spike bull: {int(pattern_counts[8])}")
        print(f"    CVD spike bear: {int(pattern_counts[9])}")
        print(f"  Basis Patterns:")
        print(f"    Basis long: {int(pattern_counts[10])}")
        print(f"    Basis short: {int(pattern_counts[11])}")
        print(f"  OI Patterns:")
        print(f"    Accumulation: {int(pattern_counts[12])}")
        print(f"    Distribution: {int(pattern_counts[13])}")
        print(f"  Volume Profile Patterns:")
        print(f"    POC breakout up: {int(pattern_counts[14])}")
        print(f"    POC breakout down: {int(pattern_counts[15])}")
        print(f"  No pattern (hold): {int(pattern_counts[16])}")
    
    def _add_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Profile indicators."""
        poc_list = []
        vah_list = []
        val_list = []
        
        for i in range(len(df)):
            if i < self.volume_profile_period:
                poc_list.append(df['spot_close'].iloc[i])
                vah_list.append(df['spot_high'].iloc[i])
                val_list.append(df['spot_low'].iloc[i])
                continue
            
            period_df = df.iloc[i - self.volume_profile_period:i]
            
            price_min = period_df['spot_low'].min()
            price_max = period_df['spot_high'].max()
            num_bins = 50
            bins = np.linspace(price_min, price_max, num_bins + 1)
            
            volume_at_price = np.zeros(num_bins)
            
            for _, row in period_df.iterrows():
                low_idx = np.digitize(row['spot_low'], bins) - 1
                high_idx = np.digitize(row['spot_high'], bins) - 1
                low_idx = max(0, min(low_idx, num_bins - 1))
                high_idx = max(0, min(high_idx, num_bins - 1))
                
                if high_idx >= low_idx:
                    volume_per_bin = row['spot_volume'] / (high_idx - low_idx + 1)
                    volume_at_price[low_idx:high_idx + 1] += volume_per_bin
            
            poc_idx = np.argmax(volume_at_price)
            poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
            
            total_volume = volume_at_price.sum()
            target_volume = total_volume * 0.70
            
            cumsum_volume = volume_at_price[poc_idx]
            val_idx = poc_idx
            vah_idx = poc_idx
            
            while cumsum_volume < target_volume and (val_idx > 0 or vah_idx < num_bins - 1):
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
        
        df['distance_to_poc'] = (df['spot_close'] - df['poc']) / df['spot_close']
        df['distance_to_vah'] = (df['spot_close'] - df['vah']) / df['spot_close']
        df['distance_to_val'] = (df['spot_close'] - df['val']) / df['spot_close']
        
        return df
    
    def _add_basis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Basis features."""
        df['basis_change'] = df['basis'].diff()
        df['basis_pct_change'] = df['basis_pct'].diff()
        df['basis_zscore'] = ((df['basis'] - df['basis'].rolling(100).mean()) / 
                              (df['basis'].rolling(100).std() + 1e-8))
        df['basis_trend'] = df['basis'].diff(5)
        df['basis_extreme'] = np.abs(df['basis_zscore'])
        
        return df
    
    def _add_open_interest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Open Interest features."""
        if 'open_interest' not in df.columns:
            df['oi_change'] = 0
            df['oi_pct_change'] = 0
            df['oi_zscore'] = 0
            df['oi_trend'] = 0
            return df
        
        df['oi_change'] = df['open_interest'].diff()
        df['oi_pct_change'] = df['open_interest'].pct_change()
        df['oi_zscore'] = ((df['open_interest'] - df['open_interest'].rolling(100).mean()) / 
                           (df['open_interest'].rolling(100).std() + 1e-8))
        df['oi_trend'] = df['open_interest'].diff(5)
        
        return df
    
    def _add_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        df['volume_ratio'] = df['futures_volume'] / (df['spot_volume'] + 1e-8)
        df['cvd_divergence'] = df['futures_cvd'] - df['spot_cvd']
        
        df['spot_returns'] = df['spot_close'].pct_change()
        df['futures_returns'] = df['futures_close'].pct_change()
        df['spot_volatility'] = df['spot_returns'].rolling(20).std()
        df['futures_volatility'] = df['futures_returns'].rolling(20).std()
        
        return df
    
    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trading patterns and label them with future profitability."""
        lookback = 14
        
        # Calculate future returns for pattern validation
        future_price = df['spot_close'].shift(-self.prediction_horizon)
        future_return = (future_price - df['spot_close']) / df['spot_close']
        
        # Price and indicator changes
        price_change = df['spot_close'].diff(lookback)
        spot_cvd_change = df['spot_cvd'].diff(lookback)
        futures_cvd_change = df['futures_cvd'].diff(lookback)
        oi_change = df['open_interest'].diff(lookback) if 'open_interest' in df.columns else 0
        
        # CVD momentum and trends
        spot_cvd_momentum = df['spot_cvd'].diff(5)
        futures_cvd_momentum = df['futures_cvd'].diff(5)
        spot_cvd_accel = spot_cvd_momentum.diff(3)  # Acceleration
        futures_cvd_accel = futures_cvd_momentum.diff(3)
        
        # Volume spikes
        volume_spike = df['spot_volume'] > df['spot_volume'].rolling(20).mean() * 1.5
        
        # === CVD PATTERNS ===
        
        # Pattern 1: Bullish Divergence (price down, CVD up)
        bullish_div = ((price_change < -100) & (spot_cvd_change > 0) & 
                       (future_return > self.min_pattern_profit))
        
        # Pattern 2: Bearish Divergence (price up, CVD down)
        bearish_div = ((price_change > 100) & (spot_cvd_change < 0) & 
                       (future_return < -self.min_pattern_profit))
        
        # Pattern 3: CVD Momentum Reversal Bullish (CVD was falling, now rising strongly)
        cvd_reversal_bull = ((spot_cvd_momentum.shift(5) < 0) & 
                             (spot_cvd_momentum > 0) & 
                             (spot_cvd_accel > 0) &
                             (future_return > self.min_pattern_profit))
        
        # Pattern 4: CVD Momentum Reversal Bearish (CVD was rising, now falling strongly)
        cvd_reversal_bear = ((spot_cvd_momentum.shift(5) > 0) & 
                             (spot_cvd_momentum < 0) & 
                             (spot_cvd_accel < 0) &
                             (future_return < -self.min_pattern_profit))
        
        # Pattern 5: CVD Exhaustion Bullish (CVD extremely negative, reversal coming)
        spot_cvd_zscore = ((df['spot_cvd'] - df['spot_cvd'].rolling(100).mean()) / 
                           (df['spot_cvd'].rolling(100).std() + 1e-8))
        cvd_exhaustion_bull = ((spot_cvd_zscore < -2.5) & 
                               (spot_cvd_momentum > 0) &
                               (future_return > self.min_pattern_profit))
        
        # Pattern 6: CVD Exhaustion Bearish (CVD extremely positive, reversal coming)
        cvd_exhaustion_bear = ((spot_cvd_zscore > 2.5) & 
                               (spot_cvd_momentum < 0) &
                               (future_return < -self.min_pattern_profit))
        
        # Pattern 7: Spot-Futures CVD Divergence Bullish (spot CVD up, futures CVD down)
        cvd_spot_futures_bull = ((spot_cvd_change > 0) & 
                                 (futures_cvd_change < 0) &
                                 (future_return > self.min_pattern_profit))
        
        # Pattern 8: Spot-Futures CVD Divergence Bearish (spot CVD down, futures CVD up)
        cvd_spot_futures_bear = ((spot_cvd_change < 0) & 
                                 (futures_cvd_change > 0) &
                                 (future_return < -self.min_pattern_profit))
        
        # Pattern 9: CVD Spike with Volume (strong buying/selling pressure)
        cvd_spike_bull = ((spot_cvd_momentum > spot_cvd_momentum.rolling(20).mean() + 2 * spot_cvd_momentum.rolling(20).std()) &
                          volume_spike &
                          (future_return > self.min_pattern_profit))
        
        cvd_spike_bear = ((spot_cvd_momentum < spot_cvd_momentum.rolling(20).mean() - 2 * spot_cvd_momentum.rolling(20).std()) &
                          volume_spike &
                          (future_return < -self.min_pattern_profit))
        
        # === BASIS PATTERNS ===
        
        # Pattern 10: Basis Long (basis too negative, will normalize)
        basis_long = ((df['basis_zscore'] < -2.0) & 
                      (future_return > self.min_pattern_profit))
        
        # Pattern 11: Basis Short (basis too positive, will normalize)
        basis_short = ((df['basis_zscore'] > 2.0) & 
                       (future_return < -self.min_pattern_profit))
        
        # === OI PATTERNS ===
        
        # Pattern 12: Accumulation (OI rising, price flat, then up)
        price_flat = np.abs(df['spot_close'].pct_change(5)) < 0.01
        oi_rising = df['oi_trend'] > 0 if 'open_interest' in df.columns else False
        accumulation = (price_flat & oi_rising & (future_return > self.min_pattern_profit))
        
        # Pattern 13: Distribution (OI rising, price flat, then down)
        distribution = (price_flat & oi_rising & (future_return < -self.min_pattern_profit))
        
        # === VOLUME PROFILE PATTERNS ===
        
        # Pattern 14: POC Breakout Up (price breaks above POC with volume)
        poc_break_up = ((df['spot_close'] > df['poc']) & 
                        (df['spot_close'].shift(1) <= df['poc'].shift(1)) &
                        volume_spike & (future_return > self.min_pattern_profit))
        
        # Pattern 15: POC Breakout Down
        poc_break_down = ((df['spot_close'] < df['poc']) & 
                          (df['spot_close'].shift(1) >= df['poc'].shift(1)) &
                          volume_spike & (future_return < -self.min_pattern_profit))
        
        # Store all patterns
        df['pattern_bullish_div'] = bullish_div.astype(float)
        df['pattern_bearish_div'] = bearish_div.astype(float)
        df['pattern_cvd_reversal_bull'] = cvd_reversal_bull.astype(float)
        df['pattern_cvd_reversal_bear'] = cvd_reversal_bear.astype(float)
        df['pattern_cvd_exhaustion_bull'] = cvd_exhaustion_bull.astype(float)
        df['pattern_cvd_exhaustion_bear'] = cvd_exhaustion_bear.astype(float)
        df['pattern_cvd_spot_futures_bull'] = cvd_spot_futures_bull.astype(float)
        df['pattern_cvd_spot_futures_bear'] = cvd_spot_futures_bear.astype(float)
        df['pattern_cvd_spike_bull'] = cvd_spike_bull.astype(float)
        df['pattern_cvd_spike_bear'] = cvd_spike_bear.astype(float)
        df['pattern_basis_long'] = basis_long.astype(float)
        df['pattern_basis_short'] = basis_short.astype(float)
        df['pattern_accumulation'] = accumulation.astype(float)
        df['pattern_distribution'] = distribution.astype(float)
        df['pattern_poc_break_up'] = poc_break_up.astype(float)
        df['pattern_poc_break_down'] = poc_break_down.astype(float)
        
        # Pattern 16: No pattern (hold)
        has_pattern = (bullish_div | bearish_div | cvd_reversal_bull | cvd_reversal_bear |
                       cvd_exhaustion_bull | cvd_exhaustion_bear | cvd_spot_futures_bull | cvd_spot_futures_bear |
                       cvd_spike_bull | cvd_spike_bear | basis_long | basis_short | 
                       accumulation | distribution | poc_break_up | poc_break_down)
        df['pattern_hold'] = (~has_pattern).astype(float)
        
        # Store future return for training
        df['future_return'] = future_return
        
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features with Z-score normalization."""
        window = 100
        
        # Price features - RELATIVE
        df['spot_price_zscore'] = ((df['spot_close'] - df['spot_close'].rolling(window).mean()) / 
                                   (df['spot_close'].rolling(window).std() + 1e-8)).fillna(0)
        df['futures_price_zscore'] = ((df['futures_close'] - df['futures_close'].rolling(window).mean()) / 
                                      (df['futures_close'].rolling(window).std() + 1e-8)).fillna(0)
        
        # Volume - RELATIVE
        df['spot_volume_zscore'] = ((df['spot_volume'] - df['spot_volume'].rolling(window).mean()) / 
                                    (df['spot_volume'].rolling(window).std() + 1e-8)).fillna(0)
        df['futures_volume_zscore'] = ((df['futures_volume'] - df['futures_volume'].rolling(window).mean()) / 
                                       (df['futures_volume'].rolling(window).std() + 1e-8)).fillna(0)
        
        # CVD - RELATIVE
        df['spot_cvd_zscore'] = ((df['spot_cvd'] - df['spot_cvd'].rolling(window).mean()) / 
                                 (df['spot_cvd'].rolling(window).std() + 1e-8)).fillna(0)
        df['futures_cvd_zscore'] = ((df['futures_cvd'] - df['futures_cvd'].rolling(window).mean()) / 
                                    (df['futures_cvd'].rolling(window).std() + 1e-8)).fillna(0)
        
        feature_cols = [
            # Price
            'spot_price_zscore', 'futures_price_zscore',
            'spot_returns', 'futures_returns',
            
            # Volume
            'spot_volume_zscore', 'futures_volume_zscore',
            'volume_ratio',
            
            # Volume Profile
            'distance_to_poc', 'distance_to_vah', 'distance_to_val',
            
            # Basis (CRITICAL!)
            'basis_pct', 'basis_change', 'basis_pct_change',
            'basis_zscore', 'basis_trend', 'basis_extreme',
            
            # Open Interest (CRITICAL!)
            'oi_pct_change', 'oi_zscore', 'oi_trend',
            
            # CVD
            'spot_cvd_zscore', 'futures_cvd_zscore',
            'cvd_divergence',
            
            # Volatility
            'spot_volatility', 'futures_volatility',
        ]
        
        features = df[feature_cols].fillna(0).values
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
        
        return features
    
    def _extract_pattern_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Extract pattern labels (multi-label classification)."""
        pattern_cols = [
            'pattern_bullish_div',
            'pattern_bearish_div',
            'pattern_cvd_reversal_bull',
            'pattern_cvd_reversal_bear',
            'pattern_cvd_exhaustion_bull',
            'pattern_cvd_exhaustion_bear',
            'pattern_cvd_spot_futures_bull',
            'pattern_cvd_spot_futures_bear',
            'pattern_cvd_spike_bull',
            'pattern_cvd_spike_bear',
            'pattern_basis_long',
            'pattern_basis_short',
            'pattern_accumulation',
            'pattern_distribution',
            'pattern_poc_break_up',
            'pattern_poc_break_down',
            'pattern_hold',
        ]
        
        patterns = df[pattern_cols].values
        return patterns
    
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.prediction_horizon
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx:idx + self.seq_len]
        patterns = self.patterns[idx + self.seq_len]
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(patterns, dtype=torch.float32),
        )
