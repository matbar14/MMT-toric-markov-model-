#!/usr/bin/env python3
"""Backtesting script for trading model V3 with Basis and Open Interest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--position-size", type=float, default=0.95, help="Fraction of capital per trade")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost (0.1%)")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Min confidence to trade (0.0-1.0)")
    parser.add_argument("--pattern-prob-threshold", type=float, default=0.35, help="Min pattern probability to treat signal as valid")
    parser.add_argument("--signal-threshold", type=float, default=0.28, help="Min (probability * confidence) score for entry")
    parser.add_argument("--cooldown-bars", type=int, default=4, help="Bars to wait after closing position")
    parser.add_argument("--max-hold-bars", type=int, default=96, help="Force-close position after N bars (0 to disable)")
    parser.add_argument("--take-profit", type=float, default=0.02, help="Take profit percentage (default 2%)")
    parser.add_argument("--stop-loss", type=float, default=0.01, help="Stop loss percentage (default 1%)")
    parser.add_argument("--output", type=str, default="backtest_results_v3.csv", help="Output CSV path")
    return parser.parse_args()


class TradingBacktest:
    """Backtesting engine for trading model."""
    
    def __init__(
        self,
        model: ToricTradingModelV3,
        initial_capital: float = 10000,
        position_size: float = 0.95,
        transaction_cost: float = 0.001,
        confidence_threshold: float = 0.55,
        pattern_prob_threshold: float = 0.35,
        signal_threshold: float = 0.28,
        cooldown_bars: int = 4,
        max_hold_bars: int = 96,
        take_profit_pct: float = 0.02,  # 2% take profit
        stop_loss_pct: float = 0.01,    # 1% stop loss
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.confidence_threshold = confidence_threshold
        self.pattern_prob_threshold = pattern_prob_threshold
        self.signal_threshold = signal_threshold
        self.cooldown_bars = cooldown_bars
        self.max_hold_bars = max_hold_bars
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        
        self.reset()
    
    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.position = 0  # Number of units held
        self.position_value = 0
        self.entry_price = 0  # Track entry price for TP/SL
        self.position_bars = 0
        self.cooldown_remaining = 0
        self.trades = []
        self.portfolio_values = []
    
    def check_exit_conditions(self, current_price: float) -> tuple[bool, str]:
        """Check if we should exit position based on TP/SL.
        
        Returns:
            (should_exit, reason)
        """
        if self.position == 0:
            return False, ""

        if self.max_hold_bars > 0 and self.position_bars >= self.max_hold_bars:
            return True, "MAX_HOLD"
        
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        # Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, "TAKE_PROFIT"
        
        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, "STOP_LOSS"
        
        return False, ""
    
    def execute_trade(self, action: int, price: float, timestamp: str, pattern_name: str = "", exit_reason: str = ""):
        """
        Execute a trade with TP/SL logic.
        
        Args:
            action: 0=sell, 1=hold, 2=buy
            price: Current price
            timestamp: Current timestamp
            pattern_name: Name of detected pattern
            exit_reason: Reason for exit (TP/SL/PATTERN)
        """
        if action == 2:  # Buy signal
            if self.position == 0:  # Only buy if not holding
                amount_to_invest = self.capital * self.position_size
                cost = amount_to_invest * (1 + self.transaction_cost)
                
                if cost <= self.capital:
                    self.position = amount_to_invest / price
                    self.capital -= cost
                    self.position_value = amount_to_invest
                    self.entry_price = price  # Track entry price
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'pattern': pattern_name,
                        'price': price,
                        'amount': self.position,
                        'cost': cost,
                        'capital': self.capital,
                    })
        
        elif action == 0:  # Sell signal
            if self.position > 0:  # Only sell if holding
                revenue = self.position * price * (1 - self.transaction_cost)
                pnl = revenue - (self.position * self.entry_price)
                pnl_pct = (price - self.entry_price) / self.entry_price * 100
                
                self.capital += revenue
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'pattern': pattern_name,
                    'exit_reason': exit_reason,
                    'price': price,
                    'entry_price': self.entry_price,
                    'amount': self.position,
                    'revenue': revenue,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital': self.capital,
                })
                
                self.position = 0
                self.position_value = 0
                self.entry_price = 0
                self.position_bars = 0
                self.cooldown_remaining = self.cooldown_bars
        
        # Update portfolio value
        total_value = self.capital + (self.position * price if self.position > 0 else 0)
        self.portfolio_values.append({
            'timestamp': timestamp,
            'capital': self.capital,
            'position': self.position,
            'position_value': self.position * price if self.position > 0 else 0,
            'total_value': total_value,
            'price': price,
        })
    
    def run(self, dataset: TradingDatasetV3, prices: pd.Series, timestamps: pd.Series):
        """Run backtest on dataset with PATTERN DETECTION."""
        self.reset()
        device = next(self.model.parameters()).device
        
        print(f"Running backtest on {len(dataset)} samples...")
        print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        print(f"Pattern probability threshold: {self.pattern_prob_threshold:.2f}")
        print(f"Signal score threshold: {self.signal_threshold:.2f}")
        print(f"Cooldown bars after exit: {self.cooldown_bars}")
        if self.max_hold_bars > 0:
            print(f"Max hold bars: {self.max_hold_bars}")
        
        skipped_trades = 0
        skipped_cooldown = 0
        skipped_bearish = 0
        skipped_low_score = 0
        pattern_names = [
            'Bullish Div', 'Bearish Div', 'CVD Rev Bull', 'CVD Rev Bear',
            'CVD Exh Bull', 'CVD Exh Bear', 'CVD SF Bull', 'CVD SF Bear',
            'CVD Spike Bull', 'CVD Spike Bear', 'Basis Long', 'Basis Short',
            'Accumulation', 'Distribution', 'POC Break Up', 'POC Break Down', 'Hold'
        ]
        bullish_patterns = {0, 2, 4, 6, 8, 10, 12, 14}
        
        for i in range(len(dataset)):
            # Get current price and timestamp
            price_idx = i + dataset.seq_len
            if price_idx >= len(prices):
                break
            
            price = prices.iloc[price_idx]
            timestamp = timestamps.iloc[price_idx]

            if self.position > 0:
                self.position_bars += 1
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            
            # FIRST: Check TP/SL if we have position
            should_exit, exit_reason = self.check_exit_conditions(price)
            if should_exit:
                self.execute_trade(0, price, timestamp, "", exit_reason)
                if (i + 1) % 100 == 0:
                    current_value = self.portfolio_values[-1]['total_value']
                    num_trades = len(self.trades)
                    print(f"  Step {i + 1}/{len(dataset)}: "
                          f"Portfolio=${current_value:.2f}, "
                          f"Return={100 * (current_value / self.initial_capital - 1):.2f}%, "
                          f"Trades={num_trades}, "
                          f"Position={'LONG' if self.position > 0 else 'FLAT'}")
                    print(f"    Exit: {exit_reason}")
                continue
            
            # SECOND: Check for new pattern entry (only if flat)
            if self.position == 0:
                features, _ = dataset[i]
                features = features.unsqueeze(0).to(device)
                
                # Detect patterns
                with torch.no_grad():
                    result = self.model.detect_patterns(
                        features,
                        confidence_threshold=self.confidence_threshold,
                        pattern_prob_threshold=self.pattern_prob_threshold,
                    )

                    best_pattern = result['best_non_hold_pattern'].item()
                    best_prob = result['best_non_hold_prob'].item()
                    best_confidence = result['best_non_hold_confidence'].item()
                    best_score = result['best_non_hold_score'].item()
                    hold_score = result['hold_score'].item()
                    has_pattern = result['has_pattern'].item()
                
                # Determine action based on pattern
                action = 1  # Default: hold
                pattern_name = "No pattern"

                if self.cooldown_remaining > 0:
                    skipped_cooldown += 1
                elif has_pattern:
                    pattern_name = pattern_names[best_pattern]
                    if best_pattern in bullish_patterns:
                        if best_score >= self.signal_threshold:
                            action = 2  # Buy
                        else:
                            skipped_low_score += 1
                    else:
                        skipped_bearish += 1
                else:
                    skipped_trades += 1
                
                # Execute trade
                self.execute_trade(action, price, timestamp, pattern_name, "")
                
                if (i + 1) % 100 == 0:
                    current_value = self.portfolio_values[-1]['total_value']
                    num_trades = len(self.trades)
                    print(f"  Step {i + 1}/{len(dataset)}: "
                          f"Portfolio=${current_value:.2f}, "
                          f"Return={100 * (current_value / self.initial_capital - 1):.2f}%, "
                          f"Trades={num_trades}, "
                          f"Position={'LONG' if self.position > 0 else 'FLAT'}")
                    if has_pattern:
                        print(
                            f"    Last pattern: {pattern_name} "
                            f"(prob={best_prob:.2f}, conf={best_confidence:.2f}, score={best_score:.2f}, hold_score={hold_score:.2f})"
                        )
            else:
                # Just update portfolio value if holding
                total_value = self.capital + (self.position * price)
                self.portfolio_values.append({
                    'timestamp': timestamp,
                    'capital': self.capital,
                    'position': self.position,
                    'position_value': self.position * price,
                    'total_value': total_value,
                    'price': price,
                })
                
                if (i + 1) % 100 == 0:
                    current_value = self.portfolio_values[-1]['total_value']
                    num_trades = len(self.trades)
                    pnl_pct = (price - self.entry_price) / self.entry_price * 100
                    print(f"  Step {i + 1}/{len(dataset)}: "
                          f"Portfolio=${current_value:.2f}, "
                          f"Return={100 * (current_value / self.initial_capital - 1):.2f}%, "
                          f"Trades={num_trades}, "
                          f"Position=LONG (PnL: {pnl_pct:+.2f}%)")
        
        print(f"\nTotal patterns detected: {len(dataset) - skipped_trades}/{len(dataset)}")
        print(f"Skipped by cooldown: {skipped_cooldown}")
        print(f"Skipped bearish patterns: {skipped_bearish}")
        print(f"Skipped low-score bullish patterns: {skipped_low_score}")
        print(f"Total trades executed: {len(self.trades)}")
        return self.get_metrics()
    
    def get_metrics(self) -> dict:
        """Calculate performance metrics."""
        if not self.portfolio_values:
            return {}
        
        values = [pv['total_value'] for pv in self.portfolio_values]
        returns = np.diff(values) / values[:-1]
        
        final_value = values[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Sharpe ratio (assuming 252 trading days per year)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        num_trades = len(self.trades)
        if num_trades > 0:
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            
            wins = 0
            for buy, sell in zip(buy_trades, sell_trades):
                if sell['price'] > buy['price']:
                    wins += 1
            
            win_rate = 100 * wins / len(sell_trades) if sell_trades else 0
        else:
            win_rate = 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'num_trades': num_trades,
            'win_rate_pct': win_rate,
        }
        
        return metrics


def main():
    args = parse_args()
    device = select_device("cuda")
    
    print("=" * 80)
    print("BACKTEST V3 MODEL WITH PATTERN DETECTION")
    print("=" * 80)
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint["args"]

    normalization = checkpoint.get("normalization")
    norm_stats = None
    if normalization is not None:
        norm_stats = {
            "feature_mean": normalization["feature_mean"].detach().cpu().numpy(),
            "feature_std": normalization["feature_std"].detach().cpu().numpy(),
        }
    
    print("Creating V3 PATTERN DETECTION model...")
    
    # Load test dataset to get number of features
    test_dataset = TradingDatasetV3(
        csv_path=args.data,
        seq_len=ckpt_args.seq_len,
        prediction_horizon=ckpt_args.prediction_horizon,
        train=False,
        min_pattern_profit=getattr(ckpt_args, "min_pattern_profit", 0.005),
        normalization_stats=norm_stats,
    )
    num_features = test_dataset.features.shape[1]
    
    print(f"Model expects {num_features} features")
    
    model = ToricTradingModelV3(
        num_features=num_features,
        dim_angles=ckpt_args.dim_angles,
        max_len=ckpt_args.seq_len,
        num_states=ckpt_args.num_states,
        num_levels=4,
        num_layers=ckpt_args.num_layers,
        n_bits=8,
        use_attention=True,
        num_patterns=17,  # 16 patterns + hold
        predict_return=True,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}, "
          f"val_acc={checkpoint['val_accuracy']:.2f}%)")
    if "val_non_hold_f1" in checkpoint:
        print(
            f"Checkpoint non-hold F1={checkpoint['val_non_hold_f1']:.2f}% "
            f"(P={checkpoint.get('val_non_hold_precision', 0.0):.2f}%, "
            f"R={checkpoint.get('val_non_hold_recall', 0.0):.2f}%)"
        )
    
    # Load prices and timestamps
    df = pd.read_csv(args.data)
    prices = df['spot_close']  # Use spot price for trading
    timestamps = df['timestamp'] if 'timestamp' in df.columns else pd.Series(range(len(df)))
    
    print("\nRunning backtest...")
    print("=" * 80)
    
    backtest = TradingBacktest(
        model=model,
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        transaction_cost=args.transaction_cost,
        confidence_threshold=args.confidence_threshold,
        pattern_prob_threshold=args.pattern_prob_threshold,
        signal_threshold=args.signal_threshold,
        cooldown_bars=args.cooldown_bars,
        max_hold_bars=args.max_hold_bars,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
    )
    
    metrics = backtest.run(test_dataset, prices, timestamps)
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Initial Capital:    ${metrics['initial_capital']:,.2f}")
    print(f"Final Value:        ${metrics['final_value']:,.2f}")
    print(f"Total Return:       {metrics['total_return_pct']:+.2f}%")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"Number of Trades:   {metrics['num_trades']}")
    print(f"Win Rate:           {metrics['win_rate_pct']:.2f}%")
    print("=" * 80)
    
    # Save results
    portfolio_df = pd.DataFrame(backtest.portfolio_values)
    portfolio_df.to_csv(args.output, index=False)
    print(f"\nPortfolio history saved to: {args.output}")
    
    trades_df = pd.DataFrame(backtest.trades)
    if len(trades_df) > 0:
        trades_output = args.output.replace('.csv', '_trades.csv')
        trades_df.to_csv(trades_output, index=False)
        print(f"Trade history saved to: {trades_output}")
    
    # Compare with buy-and-hold
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]
    if len(test_df) > 0:
        first_price = test_df['spot_close'].iloc[0]
        last_price = test_df['spot_close'].iloc[-1]
        bh_return = 100 * (last_price / first_price - 1)
        print(f"\nBuy-and-Hold Return: {bh_return:+.2f}%")
        print(f"Model vs B&H:        {metrics['total_return_pct'] - bh_return:+.2f}%")


if __name__ == "__main__":
    main()
