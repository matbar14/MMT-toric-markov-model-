"""Trading dataset V3 ENHANCED with temporal windows.

IMPROVEMENTS:
- Rolling statistics (mean, std, min, max) for key features over 3, 5, 10 bars
- Lag features (t-1, t-2, t-3) for price and CVD
- Trend features (slope, acceleration) for CVD and Basis
- More robust temporal context for pattern detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3


class TradingDatasetV3Enhanced(TradingDatasetV3):
    """Enhanced dataset with temporal windows for better pattern separation."""
    
    def __init__(
        self,
        csv_path: str,
        seq_len: int = 64,
        prediction_horizon: int = 4,
        train: bool = True,
        train_split: float = 0.8,
        volume_profile_period: int = 20,
        min_pattern_profit: float = 0.003,
        normalization_stats: dict[str, np.ndarray] | None = None,
        aux_target_stats: dict[str, np.ndarray] | None = None,
        return_aux_targets: bool = False,
        verbose: bool = True,
        # New parameters for temporal features
        rolling_windows: list[int] | None = None,
        lag_periods: list[int] | None = None,
        split: str | None = None,
        validation_split: float = 0.1,
        split_boundaries: dict[str, str] | None = None,
    ):
        self.rolling_windows = rolling_windows or [3, 5, 10]
        self.lag_periods = lag_periods or [1, 2, 3]
        
        # Call parent constructor
        super().__init__(
            csv_path=csv_path,
            seq_len=seq_len,
            prediction_horizon=prediction_horizon,
            train=train,
            train_split=train_split,
            volume_profile_period=volume_profile_period,
            min_pattern_profit=min_pattern_profit,
            normalization_stats=normalization_stats,
            aux_target_stats=aux_target_stats,
            return_aux_targets=return_aux_targets,
            verbose=verbose,
            split=split,
            validation_split=validation_split,
            split_boundaries=split_boundaries,
        )
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features with temporal windows and lags."""
        # Get base features from parent
        base_features = super()._extract_features(df)
        
        # Add temporal features
        temporal_features = self._add_temporal_features(df)
        
        # Concatenate
        features = np.concatenate([base_features, temporal_features], axis=1)
        
        return features
    
    def _add_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """Add rolling statistics and lag features."""
        temporal_cols = []
        
        # Key features to add temporal context
        key_features = {
            'spot_cvd': 'cvd',
            'futures_cvd': 'futures_cvd',
            'basis_pct': 'basis',
            'spot_volume': 'volume',
            'spot_close': 'price',
            'open_interest': 'oi',
        }
        
        for col, prefix in key_features.items():
            if col not in df.columns:
                continue
            
            # Rolling statistics
            for window in self.rolling_windows:
                # Mean
                col_name = f'{prefix}_roll{window}_mean'
                df[col_name] = df[col].rolling(window, min_periods=1).mean()
                temporal_cols.append(col_name)
                
                # Std (normalized by mean)
                col_name = f'{prefix}_roll{window}_std'
                roll_mean = df[col].rolling(window, min_periods=1).mean()
                roll_std = df[col].rolling(window, min_periods=1).std()
                df[col_name] = roll_std / (np.abs(roll_mean) + 1e-8)
                temporal_cols.append(col_name)
                
                # Min/Max range (normalized)
                col_name = f'{prefix}_roll{window}_range'
                roll_min = df[col].rolling(window, min_periods=1).min()
                roll_max = df[col].rolling(window, min_periods=1).max()
                df[col_name] = (roll_max - roll_min) / (np.abs(roll_mean) + 1e-8)
                temporal_cols.append(col_name)
                
                # Current value vs rolling mean (z-score like)
                col_name = f'{prefix}_roll{window}_zscore'
                df[col_name] = (df[col] - roll_mean) / (roll_std + 1e-8)
                temporal_cols.append(col_name)
        
        # Lag features for price and CVD (most important for patterns)
        lag_features = {
            'spot_close': 'price',
            'spot_cvd': 'cvd',
            'basis_pct': 'basis',
        }
        
        for col, prefix in lag_features.items():
            if col not in df.columns:
                continue
            
            for lag in self.lag_periods:
                # Absolute lag
                col_name = f'{prefix}_lag{lag}'
                df[col_name] = df[col].shift(lag)
                temporal_cols.append(col_name)
                
                # Change from lag
                col_name = f'{prefix}_change_from_lag{lag}'
                df[col_name] = (df[col] - df[col].shift(lag)) / (np.abs(df[col].shift(lag)) + 1e-8)
                temporal_cols.append(col_name)
        
        # Trend features (slope over different windows)
        trend_features = {
            'spot_cvd': 'cvd',
            'basis_pct': 'basis',
            'open_interest': 'oi',
        }
        
        for col, prefix in trend_features.items():
            if col not in df.columns:
                continue
            
            for window in [3, 5, 10]:
                # Linear regression slope (normalized)
                col_name = f'{prefix}_slope{window}'
                slopes = []
                for i in range(len(df)):
                    if i < window:
                        slopes.append(0.0)
                    else:
                        y = df[col].iloc[i-window:i].values
                        x = np.arange(window)
                        # Simple slope: (y[-1] - y[0]) / window
                        slope = (y[-1] - y[0]) / (window * (np.abs(y[0]) + 1e-8))
                        slopes.append(slope)
                df[col_name] = slopes
                temporal_cols.append(col_name)
                
                # Acceleration (change in slope)
                if window == 5:
                    col_name = f'{prefix}_accel'
                    df[col_name] = df[f'{prefix}_slope{window}'].diff()
                    temporal_cols.append(col_name)
        
        # CVD divergence strength over time
        for window in [3, 5]:
            col_name = f'cvd_div_strength_{window}'
            price_change = df['spot_close'].pct_change(window)
            cvd_change = df['spot_cvd'].diff(window) / (df['spot_cvd'].shift(window).abs() + 1e-8)
            # Divergence: price and CVD move in opposite directions
            df[col_name] = -price_change * cvd_change  # Negative correlation
            temporal_cols.append(col_name)
        
        # Basis momentum
        for window in [3, 5]:
            col_name = f'basis_momentum_{window}'
            df[col_name] = df['basis_pct'].diff(window)
            temporal_cols.append(col_name)
        
        # Extract temporal features
        self.feature_names += temporal_cols
        features = df[temporal_cols].fillna(0).values
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
        
        return features
