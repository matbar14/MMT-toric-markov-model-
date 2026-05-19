"""Train LightGBM baseline for gate (Hold vs Any pattern).

This serves as a baseline to check if the gate task is learnable with current features.
If LightGBM achieves good precision/recall, the problem is learnable and neural network
should be able to learn it too. If not, we need better features or different problem formulation.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from toric_markov_model.data.trading_dataset_v3_enhanced import TradingDatasetV3Enhanced


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM gate baseline")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data")
    parser.add_argument("--output-dir", type=str, default="lgbm_gate_baseline",
                        help="Output directory for model and results")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--min-pattern-profit", type=float, default=0.003)
    
    # LightGBM hyperparameters
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=0.1)
    
    # Class imbalance handling
    parser.add_argument("--scale-pos-weight", type=float, default=None,
                        help="Scale weight for positive class (auto if None)")
    
    return parser.parse_args()


def compute_metrics_at_thresholds(y_true, y_pred_proba, thresholds=None):
    """Compute precision, recall, F1 at different thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    for thr in thresholds:
        y_pred = (y_pred_proba >= thr).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        results.append({
            'threshold': float(thr),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        })
    
    return results


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("LIGHTGBM GATE BASELINE")
    print("Task: Binary classification Hold vs Any pattern")
    print("=" * 80)
    
    # Load datasets with enhanced features
    print("\nLoading enhanced datasets...")
    train_dataset = TradingDatasetV3Enhanced(
        csv_path=args.data,
        train=True,
        train_split=args.train_split,
        prediction_horizon=args.prediction_horizon,
        min_pattern_profit=args.min_pattern_profit,
        verbose=True,
    )
    
    val_dataset = TradingDatasetV3Enhanced(
        csv_path=args.data,
        train=False,
        train_split=args.train_split,
        prediction_horizon=args.prediction_horizon,
        min_pattern_profit=args.min_pattern_profit,
        normalization_stats=train_dataset.get_normalization_stats(),
        aux_target_stats=train_dataset.get_aux_target_stats(),
        verbose=True,
    )
    
    # Extract features and labels
    X_train = train_dataset.features
    y_train_patterns = train_dataset.patterns
    # Gate label: 1 if any non-hold pattern, 0 if hold
    y_train = (y_train_patterns[:, :-1].sum(axis=1) > 0).astype(int)
    
    X_val = val_dataset.features
    y_val_patterns = val_dataset.patterns
    y_val = (y_val_patterns[:, :-1].sum(axis=1) > 0).astype(int)
    
    print(f"\nTrain: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"  Positive (non-hold): {y_train.sum()} ({100 * y_train.mean():.2f}%)")
    print(f"  Negative (hold): {(1 - y_train).sum()} ({100 * (1 - y_train.mean()):.2f}%)")
    
    print(f"\nVal: {len(X_val)} samples, {X_val.shape[1]} features")
    print(f"  Positive (non-hold): {y_val.sum()} ({100 * y_val.mean():.2f}%)")
    print(f"  Negative (hold): {(1 - y_val).sum()} ({100 * (1 - y_val.mean()):.2f}%)")
    
    # Auto scale_pos_weight
    if args.scale_pos_weight is None:
        scale_pos_weight = (1 - y_train.mean()) / (y_train.mean() + 1e-8)
        print(f"\nAuto scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = args.scale_pos_weight
    
    # Train LightGBM
    print("\nTraining LightGBM...")
    print(f"  num_leaves={args.num_leaves}, max_depth={args.max_depth}")
    print(f"  learning_rate={args.learning_rate}, n_estimators={args.n_estimators}")
    print(f"  scale_pos_weight={scale_pos_weight:.2f}")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['auc', 'binary_logloss'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=10),
        ],
    )
    
    # Save model
    model_path = output_dir / "lgbm_gate.txt"
    model.booster_.save_model(str(model_path))
    print(f"\n✓ Saved model to {model_path}")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # ROC AUC
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    # Average Precision (PR AUC)
    train_ap = average_precision_score(y_train, y_train_pred_proba)
    val_ap = average_precision_score(y_val, y_val_pred_proba)
    
    print(f"\nTrain ROC AUC: {train_auc:.4f}, PR AUC: {train_ap:.4f}")
    print(f"Val   ROC AUC: {val_auc:.4f}, PR AUC: {val_ap:.4f}")
    
    # Metrics at different thresholds
    print("\nValidation metrics at different thresholds:")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 80)
    
    val_threshold_metrics = compute_metrics_at_thresholds(y_val, y_val_pred_proba)
    
    best_f1_idx = np.argmax([m['f1'] for m in val_threshold_metrics])
    best_f1_metric = val_threshold_metrics[best_f1_idx]
    
    for i, m in enumerate(val_threshold_metrics):
        marker = " ← BEST F1" if i == best_f1_idx else ""
        print(f"{m['threshold']:<12.2f} {m['precision']:<12.2%} {m['recall']:<12.2%} "
              f"{m['f1']:<12.2%} {m['tp']:<8} {m['fp']:<8} {m['fn']:<8}{marker}")
    
    print(f"\nBest F1: {best_f1_metric['f1']:.2%} at threshold {best_f1_metric['threshold']:.2f}")
    print(f"  Precision: {best_f1_metric['precision']:.2%}")
    print(f"  Recall: {best_f1_metric['recall']:.2%}")
    
    # Classification report at best threshold
    y_val_pred = (y_val_pred_proba >= best_f1_metric['threshold']).astype(int)
    print("\nClassification report at best F1 threshold:")
    print(classification_report(y_val, y_val_pred, target_names=['Hold', 'Non-Hold']))
    
    # Feature importance
    print("\nTop 20 most important features:")
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))
    
    # Save results
    results = {
        'args': vars(args),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'num_features': int(X_train.shape[1]),
        'train_positive_ratio': float(y_train.mean()),
        'val_positive_ratio': float(y_val.mean()),
        'train_auc': float(train_auc),
        'val_auc': float(val_auc),
        'train_ap': float(train_ap),
        'val_ap': float(val_ap),
        'best_threshold': float(best_f1_metric['threshold']),
        'best_f1': float(best_f1_metric['f1']),
        'best_precision': float(best_f1_metric['precision']),
        'best_recall': float(best_f1_metric['recall']),
        'threshold_metrics': val_threshold_metrics,
        'feature_importance': feature_importance.to_dict('records'),
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {results_path}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if best_f1_metric['f1'] > 0.30:
        print("✓ GOOD: F1 > 30% suggests the gate task is learnable with current features.")
        print("  Neural network should be able to achieve similar or better results.")
        if best_f1_metric['precision'] < 0.15:
            print("  ⚠ Low precision suggests many false positives.")
            print("    Consider: higher threshold, better features, or stricter pattern definitions.")
    elif best_f1_metric['f1'] > 0.20:
        print("⚠ MODERATE: F1 20-30% suggests the task is challenging but possible.")
        print("  Consider: adding more temporal features, using sequences, or ensemble methods.")
    else:
        print("✗ POOR: F1 < 20% suggests current features are insufficient.")
        print("  Recommendations:")
        print("  1. Add more temporal context (longer windows, more lags)")
        print("  2. Use sequence models (LSTM, Transformer) instead of single-point features")
        print("  3. Revisit pattern definitions - they may be too noisy or subjective")
        print("  4. Consider alternative problem formulations (regression, ranking)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
