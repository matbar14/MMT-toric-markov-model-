"""Trading dataset V3 with PATTERN DETECTION.

CRITICAL CHANGE: Instead of predicting direction, we detect PATTERNS:
- CVD patterns (10 types)
- Basis patterns (2 types)
- OI patterns (2 types)
- Volume Profile patterns (2 types)
- Hold (no pattern)

Total: 17 pattern types (multi-label classification)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TradingDatasetV3(Dataset):
    """Dataset with PATTERN DETECTION instead of direction classification."""
    STD_FLOOR = 1e-3
    
    def __init__(
        self,
        csv_path: str,
        seq_len: int = 64,
        prediction_horizon: int = 4,  # 4 hours ahead to reduce label starvation
        train: bool = True,
        train_split: float = 0.8,
        volume_profile_period: int = 20,
        min_pattern_profit: float = 0.003,  # 0.3% minimum profit to label as pattern
        normalization_stats: dict[str, np.ndarray] | None = None,
        aux_target_stats: dict[str, np.ndarray] | None = None,
        return_aux_targets: bool = False,
        verbose: bool = True,
    ):
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.volume_profile_period = volume_profile_period
        self.min_pattern_profit = min_pattern_profit
        self.return_aux_targets = return_aux_targets
        self.verbose = verbose
        
        # Load data
        df = pd.read_csv(csv_path)

        required_cols = {
            'timestamp',
            'spot_open', 'spot_high', 'spot_low', 'spot_close', 'spot_volume', 'spot_cvd',
            'futures_open', 'futures_high', 'futures_low', 'futures_close', 'futures_volume', 'futures_cvd',
            'basis', 'basis_pct',
        }
        missing_required = sorted(required_cols - set(df.columns))
        if missing_required:
            raise ValueError(
                "TradingDatasetV3 expects merged spot/futures data with basis. "
                f"Missing columns: {missing_required}"
            )
        
        # Track where OI is truly available and avoid look-ahead imputation.
        if 'open_interest' in df.columns:
            oi_available = df['open_interest'].notna()
            if 'open_interest_value' in df.columns:
                oi_available = oi_available & df['open_interest_value'].notna()
            df['oi_available'] = oi_available.astype(float)
            df['open_interest'] = df['open_interest'].ffill().fillna(0.0)
            if 'open_interest_value' in df.columns:
                df['open_interest_value'] = df['open_interest_value'].ffill().fillna(0.0)
            else:
                df['open_interest_value'] = 0.0
        else:
            # Keep pipeline deterministic even if OI is absent.
            df['open_interest'] = 0.0
            df['open_interest_value'] = 0.0
            df['oi_available'] = 0.0
        
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
        self.oi_available_ratio = float(df['oi_available'].mean()) if 'oi_available' in df.columns else 0.0

        # Keep aligned market data for backtesting to avoid index drift.
        self.timestamps = (
            df['timestamp'].reset_index(drop=True)
            if 'timestamp' in df.columns
            else pd.Series(range(len(df)))
        )
        self.spot_open = df['spot_open'].reset_index(drop=True)
        self.spot_high = df['spot_high'].reset_index(drop=True)
        self.spot_low = df['spot_low'].reset_index(drop=True)
        self.spot_close = df['spot_close'].reset_index(drop=True)
        
        # Extract features and pattern labels
        self.features = self._extract_features(df)
        self.patterns = self._extract_pattern_labels(df)
        self.aux_targets = self._extract_aux_targets(df)
        
        # Normalize features
        if normalization_stats is None:
            self.feature_mean = self.features.mean(axis=0)
            self.feature_std = self.features.std(axis=0) + 1e-8
        else:
            self.feature_mean = np.asarray(normalization_stats["feature_mean"], dtype=np.float32)
            self.feature_std = np.asarray(normalization_stats["feature_std"], dtype=np.float32)
            if self.feature_mean.shape[0] != self.features.shape[1]:
                raise ValueError(
                    "Normalization stats feature dimension mismatch: "
                    f"expected {self.features.shape[1]}, got {self.feature_mean.shape[0]}"
                )
        self.feature_std = np.maximum(self.feature_std, self.STD_FLOOR)
        self.features = (self.features - self.feature_mean) / self.feature_std
        self.features = self.features.astype(np.float32, copy=False)
        self.patterns = self.patterns.astype(np.float32, copy=False)

        if aux_target_stats is None:
            self.aux_target_mean = self.aux_targets.mean(axis=0)
            self.aux_target_std = self.aux_targets.std(axis=0) + 1e-8
        else:
            self.aux_target_mean = np.asarray(aux_target_stats["aux_target_mean"], dtype=np.float32)
            self.aux_target_std = np.asarray(aux_target_stats["aux_target_std"], dtype=np.float32)
            if self.aux_target_mean.shape[0] != self.aux_targets.shape[1]:
                raise ValueError(
                    "Aux target stats dimension mismatch: "
                    f"expected {self.aux_targets.shape[1]}, got {self.aux_target_mean.shape[0]}"
                )
        self.aux_target_std = np.maximum(self.aux_target_std, self.STD_FLOOR)
        self.aux_targets = (self.aux_targets - self.aux_target_mean) / self.aux_target_std
        self.aux_targets = np.clip(self.aux_targets, -5.0, 5.0).astype(np.float32, copy=False)
        
        # Count patterns
        if self.verbose:
            pattern_counts = self.patterns.sum(axis=0)
            print(f"{'Train' if train else 'Val'} dataset: {len(self)} samples, "
                  f"{self.features.shape[1]} features")
            print(f"  OI available: {self.oi_available_ratio * 100:.2f}% rows")
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
        """Add Volume Profile indicators: POC, VAH, VAL."""
        # Use spot data for volume profile
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
        df['value_area_position'] = (df['spot_close'] - df['val']) / (df['vah'] - df['val'] + 1e-8)

        return df
    
    def _add_basis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Basis (futures-spot spread) features - CRITICAL for trading!"""
        # Basis already calculated in download script
        # basis = futures_close - spot_close
        # basis_pct = (basis / spot_close) * 100
        
        # Basis momentum (change in spread)
        df['basis_change'] = df['basis'].diff()
        df['basis_pct_change'] = df['basis_pct'].diff()
        
        # Basis Z-score (deviation from normal spread)
        df['basis_zscore'] = ((df['basis'] - df['basis'].rolling(100).mean()) / 
                              (df['basis'].rolling(100).std() + 1e-8))
        
        # Basis trend (is spread widening or narrowing?)
        df['basis_trend'] = df['basis'].diff(5)  # 5-period trend
        
        # Basis extremes (contango/backwardation strength)
        df['basis_extreme'] = np.abs(df['basis_zscore'])
        
        return df
    
    def _add_open_interest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Open Interest features - shows real positions!"""
        oi = df['open_interest']
        oi_available = df.get('oi_available', pd.Series(0.0, index=df.index)).astype(float)
        oi_available_bool = oi_available > 0.5

        # Require contiguous OI history for derivatives to avoid false spikes.
        oi_diff_valid = oi_available_bool & oi_available_bool.shift(1, fill_value=False)
        oi_trend_valid = oi_available_bool.rolling(6, min_periods=6).sum().eq(6)
        oi_zscore_valid = oi_available_bool.rolling(100, min_periods=100).sum().eq(100)

        oi_change = oi.diff().where(oi_diff_valid, 0.0)
        oi_pct_change = oi.pct_change().replace([np.inf, -np.inf], np.nan).where(oi_diff_valid, 0.0)

        oi_zscore = ((oi - oi.rolling(100).mean()) / (oi.rolling(100).std() + 1e-8)).where(oi_zscore_valid, 0.0)
        oi_trend = oi.diff(5).where(oi_trend_valid, 0.0)
        oi_volume_ratio = (
            oi / (df['futures_volume'].rolling(20).mean() + 1e-8)
        ).where(oi_available_bool, 0.0)

        df['oi_change'] = oi_change
        df['oi_pct_change'] = oi_pct_change
        df['oi_zscore'] = oi_zscore
        df['oi_trend'] = oi_trend
        df['oi_volume_ratio'] = oi_volume_ratio
        
        return df
    
    def _add_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Spot vs Futures volume ratio
        df['volume_ratio'] = df['futures_volume'] / (df['spot_volume'] + 1e-8)
        
        # CVD divergence between spot and futures
        df['cvd_divergence'] = df['futures_cvd'] - df['spot_cvd']
        df['cvd_divergence_norm'] = df['cvd_divergence'] / (df['spot_cvd'].rolling(20).std() + 1e-8)
        
        # Price divergence (should be same as basis, but calculated differently)
        df['price_divergence'] = (df['futures_close'] - df['spot_close']) / df['spot_close']
        
        # Volatility
        df['spot_returns'] = df['spot_close'].pct_change()
        df['futures_returns'] = df['futures_close'].pct_change()
        df['spot_volatility'] = df['spot_returns'].rolling(20).std()
        df['futures_volatility'] = df['futures_returns'].rolling(20).std()
        df['volatility_ratio'] = df['futures_volatility'] / (df['spot_volatility'] + 1e-8)
        
        # VWAP
        df['spot_vwap'] = (df['spot_close'] * df['spot_volume']).cumsum() / df['spot_volume'].cumsum()
        df['vwap_distance'] = (df['spot_close'] - df['spot_vwap']) / df['spot_vwap']
        
        return df
    
    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trading patterns at the MOMENT they start (entry point).
        
        Pattern = START of movement, not prediction of future!
        We detect when conditions are met AND movement is beginning.
        """
        lookback = 14
        # Volatility-adaptive threshold makes divergence detection scale-safe
        # across different BTC price regimes and prevents label starvation.
        rolling_vol = df['spot_close'].pct_change().rolling(lookback).std().fillna(0.0)
        long_move_threshold = df['spot_close'] * rolling_vol.mul(1.5).clip(lower=0.002, upper=0.03)
        
        # Calculate CURRENT and RECENT price changes
        price_change_long = df['spot_close'].diff(lookback)  # Long-term trend
        price_change_short = df['spot_close'].diff(3)  # Short-term (beginning of move)
        
        # CVD changes
        spot_cvd_change_long = df['spot_cvd'].diff(lookback)
        spot_cvd_change_short = df['spot_cvd'].diff(3)
        futures_cvd_change_long = df['futures_cvd'].diff(lookback)
        futures_cvd_change_short = df['futures_cvd'].diff(3)
        
        # CVD momentum (current)
        spot_cvd_momentum = df['spot_cvd'].diff(2)
        futures_cvd_momentum = df['futures_cvd'].diff(2)
        
        # Volume spike (NOW)
        volume_spike = df['spot_volume'] > df['spot_volume'].rolling(20).mean() * 1.5
        
        # Future return for validation (but pattern triggers NOW)
        future_price = df['spot_close'].shift(-self.prediction_horizon)
        future_return = (future_price - df['spot_close']) / df['spot_close']
        
        # === CVD PATTERNS (detect at START of movement) ===
        
        # Pattern 1: Bullish Divergence - price fell, CVD rising, NOW starting to reverse up
        bullish_div = ((price_change_long < -long_move_threshold) &  # Price was falling
                       (spot_cvd_change_long > 0) &  # But CVD was rising (accumulation)
                       (price_change_short > 0) &  # NOW price starting to rise
                       (future_return > self.min_pattern_profit))  # And will continue
        
        # Pattern 2: Bearish Divergence - price rose, CVD falling, NOW starting to reverse down
        bearish_div = ((price_change_long > long_move_threshold) &  # Price was rising
                       (spot_cvd_change_long < 0) &  # But CVD was falling (distribution)
                       (price_change_short < 0) &  # NOW price starting to fall
                       (future_return < -self.min_pattern_profit))  # And will continue
        
        # Pattern 3: CVD Momentum Reversal Bullish - CVD just reversed up
        cvd_reversal_bull = ((spot_cvd_momentum.shift(3) < 0) &  # Was falling
                             (spot_cvd_momentum > 0) &  # NOW rising
                             (price_change_short >= 0) &  # Price confirming
                             (future_return > self.min_pattern_profit))
        
        # Pattern 4: CVD Momentum Reversal Bearish - CVD just reversed down
        cvd_reversal_bear = ((spot_cvd_momentum.shift(3) > 0) &  # Was rising
                             (spot_cvd_momentum < 0) &  # NOW falling
                             (price_change_short <= 0) &  # Price confirming
                             (future_return < -self.min_pattern_profit))
        
        # Pattern 5: CVD Exhaustion Bullish - CVD at extreme low, NOW reversing
        spot_cvd_zscore = ((df['spot_cvd'] - df['spot_cvd'].rolling(100).mean()) / 
                           (df['spot_cvd'].rolling(100).std() + 1e-8))
        cvd_exhaustion_bull = ((spot_cvd_zscore < -2.5) &  # Extreme low
                               (spot_cvd_momentum > 0) &  # NOW reversing up
                               (price_change_short > 0) &  # Price following
                               (future_return > self.min_pattern_profit))
        
        # Pattern 6: CVD Exhaustion Bearish - CVD at extreme high, NOW reversing
        cvd_exhaustion_bear = ((spot_cvd_zscore > 2.5) &  # Extreme high
                               (spot_cvd_momentum < 0) &  # NOW reversing down
                               (price_change_short < 0) &  # Price following
                               (future_return < -self.min_pattern_profit))
        
        # Pattern 7: Spot-Futures CVD Divergence Bullish - divergence happening NOW
        cvd_spot_futures_bull = ((spot_cvd_change_short > 0) &  # Spot CVD rising NOW
                                 (futures_cvd_change_short < 0) &  # Futures CVD falling NOW
                                 (price_change_short >= 0) &  # Price starting up
                                 (future_return > self.min_pattern_profit))
        
        # Pattern 8: Spot-Futures CVD Divergence Bearish
        cvd_spot_futures_bear = ((spot_cvd_change_short < 0) &  # Spot CVD falling NOW
                                 (futures_cvd_change_short > 0) &  # Futures CVD rising NOW
                                 (price_change_short <= 0) &  # Price starting down
                                 (future_return < -self.min_pattern_profit))
        
        # Pattern 9: CVD Spike with Volume - spike happening NOW
        cvd_spike_bull = ((spot_cvd_momentum > spot_cvd_momentum.rolling(20).mean() + 2 * spot_cvd_momentum.rolling(20).std()) &
                          volume_spike &  # Volume spike NOW
                          (price_change_short > 0) &  # Price moving NOW
                          (future_return > self.min_pattern_profit))
        
        cvd_spike_bear = ((spot_cvd_momentum < spot_cvd_momentum.rolling(20).mean() - 2 * spot_cvd_momentum.rolling(20).std()) &
                          volume_spike &  # Volume spike NOW
                          (price_change_short < 0) &  # Price moving NOW
                          (future_return < -self.min_pattern_profit))
        
        # === BASIS PATTERNS (mean reversion starting NOW) ===
        
        basis_change_short = df['basis'].diff(2)
        
        # Pattern 10: Basis Long - basis too negative, NOW starting to normalize
        basis_long = ((df['basis_zscore'] < -2.0) &  # Extreme negative
                      (basis_change_short > 0) &  # NOW normalizing
                      (future_return > self.min_pattern_profit))
        
        # Pattern 11: Basis Short - basis too positive, NOW starting to normalize
        basis_short = ((df['basis_zscore'] > 2.0) &  # Extreme positive
                       (basis_change_short < 0) &  # NOW normalizing
                       (future_return < -self.min_pattern_profit))
        
        # === OI PATTERNS (breakout from accumulation/distribution) ===
        
        # Pattern 12: Accumulation breakout - OI was rising, price flat, NOW breaking up
        price_was_flat = np.abs(df['spot_close'].pct_change(5).shift(2)) < 0.01
        oi_context_valid = df.get('oi_available', pd.Series(0.0, index=df.index)).rolling(6, min_periods=6).sum().eq(6)
        oi_rising = (df['oi_trend'] > 0) & oi_context_valid
        accumulation = (price_was_flat &  # Was flat
                        oi_rising &  # OI was building
                        (price_change_short > 0) &  # NOW breaking up
                        (future_return > self.min_pattern_profit))
        
        # Pattern 13: Distribution breakout - OI was rising, price flat, NOW breaking down
        distribution = (price_was_flat &  # Was flat
                        oi_rising &  # OI was building
                        (price_change_short < 0) &  # NOW breaking down
                        (future_return < -self.min_pattern_profit))
        
        # === VOLUME PROFILE PATTERNS (breakout happening NOW) ===
        
        # Pattern 14: POC Breakout Up - JUST broke above POC with volume
        poc_break_up = ((df['spot_close'] > df['poc']) &  # Above POC NOW
                        (df['spot_close'].shift(1) <= df['poc'].shift(1)) &  # Was below
                        volume_spike &  # Volume confirming
                        (future_return > self.min_pattern_profit))
        
        # Pattern 15: POC Breakout Down - JUST broke below POC with volume
        poc_break_down = ((df['spot_close'] < df['poc']) &  # Below POC NOW
                          (df['spot_close'].shift(1) >= df['poc'].shift(1)) &  # Was above
                          volume_spike &  # Volume confirming
                          (future_return < -self.min_pattern_profit))
        
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
        
        # Volume features - RELATIVE
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
            # RELATIVE PRICE
            'spot_price_zscore', 'futures_price_zscore',
            'spot_returns', 'futures_returns',
            
            # RELATIVE VOLUME
            'spot_volume_zscore', 'futures_volume_zscore',
            'volume_ratio',
            
            # VOLUME PROFILE
            'distance_to_poc', 'distance_to_vah', 'distance_to_val',
            
            # BASIS (CRITICAL!)
            'basis_pct', 'basis_change', 'basis_pct_change',
            'basis_zscore', 'basis_trend', 'basis_extreme',
            
            # OPEN INTEREST (CRITICAL!)
            'oi_pct_change', 'oi_zscore', 'oi_trend', 'oi_volume_ratio', 'oi_available',
            
            # CVD
            'spot_cvd_zscore', 'futures_cvd_zscore',
            'cvd_divergence_norm',
            
            # MARKET MICROSTRUCTURE
            'price_divergence',
            'spot_volatility', 'futures_volatility',
            'vwap_distance',
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

    def _extract_aux_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Extract normalized auxiliary regression targets."""
        horizon = self.prediction_horizon
        future_volume_change = (
            (df["futures_volume"].shift(-horizon) - df["futures_volume"])
            / (df["futures_volume"] + 1e-8)
        )
        future_cvd_change = (
            (df["futures_cvd"].shift(-horizon) - df["futures_cvd"])
            / (np.abs(df["futures_cvd"]) + 1e-8)
        )
        future_poc_movement = (
            (df["poc"].shift(-horizon) - df["poc"])
            / (df["spot_close"] + 1e-8)
        )

        aux = np.column_stack(
            [
                df["future_return"].fillna(0.0).values,
                future_volume_change.fillna(0.0).values,
                future_cvd_change.fillna(0.0).values,
                future_poc_movement.fillna(0.0).values,
            ]
        )
        aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)
        return aux
    
    def __len__(self) -> int:
        return len(self.features) - self.seq_len - self.prediction_horizon

    def get_normalization_stats(self) -> dict[str, np.ndarray]:
        return {
            "feature_mean": self.feature_mean.astype(np.float32, copy=False),
            "feature_std": self.feature_std.astype(np.float32, copy=False),
        }

    def get_aux_target_stats(self) -> dict[str, np.ndarray]:
        return {
            "aux_target_mean": self.aux_target_mean.astype(np.float32, copy=False),
            "aux_target_std": self.aux_target_std.astype(np.float32, copy=False),
        }
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx:idx + self.seq_len]
        patterns = self.patterns[idx + self.seq_len]

        output = (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(patterns, dtype=torch.float32),
        )
        if not self.return_aux_targets:
            return output

        aux = self.aux_targets[idx + self.seq_len]
        return output + (torch.tensor(aux, dtype=torch.float32),)
