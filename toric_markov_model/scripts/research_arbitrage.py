#!/usr/bin/env python3
"""Fixed chronological comparison of paired Toric/Ridge forecasts, basis rule and cash."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from toric_markov_model.arbitrage import ArbitrageConfig, ENTRY_HOURS_UTC, simulate_pairs
from toric_markov_model.data.arbitrage_download import load_inputs, write_json
from toric_markov_model.data.payoff import walk_forward_segments
from toric_markov_model.train.arbitrage import fit_pair_models, forecasts, save_pair_checkpoint, targets
from toric_markov_model.train.payoff import block_lower_mean


def passes(run, stress, minimum, max_drawdown):
    metrics = run["metrics"]
    return bool(metrics["trades"] >= minimum and metrics["trades_per_day"] >= 0.5 and
                metrics["total_return_pct"] > 0 and metrics["block_lower_daily_mean"] > 0 and
                stress["metrics"]["total_return_pct"] > 0 and
                metrics["max_drawdown_pct"] >= -100 * max_drawdown and
                stress["metrics"]["max_drawdown_pct"] >= -100 * max_drawdown)


def model_improves(run, base, stress, base_stress):
    if any(other["daily_dates"] != base["daily_dates"] for other in (run, stress, base_stress)):
        raise ValueError("paired comparison needs identical calendar dates")
    return bool(run["metrics"]["total_return_pct"] > base["metrics"]["total_return_pct"] and
                (not base["metrics"]["trades"] or
                 run["metrics"]["max_drawdown_pct"] >= base["metrics"]["max_drawdown_pct"]) and
                stress["metrics"]["total_return_pct"] > base_stress["metrics"]["total_return_pct"] and
                block_lower_mean(run["daily_returns"] - base["daily_returns"]) > 0)


def evaluate(data, entries, model, ridge):
    prediction = forecasts(model, ridge, data.windows(entries, model.max_len))
    actions = {name: (values > data.config.min_edge).astype(int) for name, values in prediction.items()}
    actions["baseline"] = data.baseline_actions(entries)
    actions["carry"] = data.carry_actions(entries)
    actions["cash"] = np.zeros(len(entries), dtype=int)
    raw = {name: simulate_pairs(data, entries, values) for name, values in actions.items()}
    stress = {name: simulate_pairs(data, entries, values, data.config.stressed()) for name, values in actions.items()}
    return prediction, actions, raw, stress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-data", type=Path, required=True)
    parser.add_argument("--arbitrage-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=30)
    for name in ("spot_fee", "futures_fee", "spot_slippage", "futures_slippage", "allocation", "margin_ratio", "min_edge", "max_drawdown"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(ArbitrageConfig(), name))
    args = parser.parse_args()
    if args.epochs < 1 or args.min_trades < 1:
        raise ValueError("positive epochs and minimum trades required")
    config = ArbitrageConfig(**{name: getattr(args, name) for name in
                                ("spot_fee", "futures_fee", "spot_slippage", "futures_slippage", "allocation", "margin_ratio", "min_edge", "max_drawdown")})
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("use a new output directory")
    data, manifest = load_inputs(args.market_data, args.arbitrage_data, config)
    segments = walk_forward_segments(len(data.frame), args.folds)
    entries_by_fold = [{name: data.entries(*bounds) for name, bounds in segment.items()} for segment in segments]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        format="arbitrage_research_v1", inputs=manifest, execution=asdict(config), segments=segments,
        entry_hours_utc=list(ENTRY_HOURS_UTC), epochs=args.epochs, min_trades=args.min_trades, seq_len=32, seed=42,
        target="net paired PnL / (spot purchase including fee + initial short collateral); same BTC quantity",
        selection="choose on selection only; independent calibration/stress; pair comparison to stronger selection-only rule",
        baseline="hypothetical full basis convergence net of four fills; last settled funding nonnegative; not guaranteed",
        carry="last settled funding rate x horizon / last observed interval, unchanged prices; heuristic, not future rate",
        boundaries="negative funding within boundary guard charged, positive boundary funding excluded",
        margin="100%+ initial collateral; conservative mark-high buffer check, not exchange liquidation simulation",
        drawdown="fixed cap on base/stress drawdown; no-worse-than-baseline additionally applies only if baseline trades",
        limitations=["hourly trade prices are not synchronized bid/ask", "no fill/hedge latency or partial fills",
                     "no true exchange margin tiers or cross-wallet transfers", "fixed six-hour exits, no price-triggered stops",
                     "single symbol, inspected data, no independent profitability proof"],
        production_approved=False, live_orders_allowed=False,
    )
    write_json(output / "protocol.json", protocol)
    folds = []
    for number, entries in enumerate(entries_by_fold, 1):
        folder = output / f"fold_{number}"
        folder.mkdir()
        model, ridge, training = fit_pair_models(data, entries["train"], entries["selection"], epochs=args.epochs)
        _, _, selection, _ = evaluate(data, entries["selection"], model, ridge)
        eligible = [name for name in ("baseline", "carry", "toric", "ridge") if
                    selection[name]["metrics"]["trades"] >= args.min_trades]
        selected = max(eligible + ["cash"], key=lambda name: (
            selection[name]["metrics"]["block_lower_daily_mean"], selection[name]["metrics"]["total_return_pct"]))
        reference = max(("baseline", "carry"), key=lambda name: (
            selection[name]["metrics"]["block_lower_daily_mean"], selection[name]["metrics"]["total_return_pct"]))
        _, _, calibration, cal_stress = evaluate(data, entries["calibration"], model, ridge)
        enabled = passes(calibration[selected], cal_stress[selected], args.min_trades, config.max_drawdown)
        if selected in ("toric", "ridge"):
            enabled &= model_improves(calibration[selected], calibration[reference],
                                      cal_stress[selected], cal_stress[reference])
        policy = dict(selected=selected, reference=reference, enabled=bool(enabled), selected_for_execution=selected if enabled else "cash",
                      selection={name: run["metrics"] for name, run in selection.items()},
                      calibration={name: run["metrics"] for name, run in calibration.items()},
                      calibration_stress={name: run["metrics"] for name, run in cal_stress.items()},
                      training=training, live_orders_allowed=False)
        save_pair_checkpoint(model, data, folder / "toric_arbitrage.pt", policy,
                             dict(inputs=manifest, segment=segments[number - 1], seed=42))
        scaler, estimator = ridge.steps[0][1], ridge.steps[1][1]
        np.savez(folder / "ridge_weights.npz", mean=scaler.mean_, scale=scaler.scale_,
                 coef=estimator.coef_, intercept=estimator.intercept_)
        policy["weights_sha256"] = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                    for name in ("toric_arbitrage.pt", "ridge_weights.npz")}
        write_json(folder / "frozen_policy.json", policy)
        prediction, actions, evaluation, stress = evaluate(data, entries["evaluation"], model, ridge)
        truth = targets(data, entries["evaluation"]).ravel()
        forecast_frame = pd.DataFrame(dict(entry=entries["evaluation"], paired_target=truth, **prediction))
        forecast_frame.to_csv(folder / "forecasts.csv", index=False)
        for name, run in evaluation.items():
            pd.DataFrame(run["trades"]).to_csv(folder / f"{name}_trades.csv", index=False)
        equity_times = data.frame.timestamp.iloc[entries["evaluation"][0]:entries["evaluation"][-1] + config.horizon]
        pd.DataFrame(dict(timestamp=equity_times, **{name: run["equity"] for name, run in evaluation.items()})).to_csv(
            folder / "equity.csv", index=False)
        result = dict(fold=number, selected=selected, reference=reference, enabled=bool(enabled),
                      frozen_policy_sha256=hashlib.sha256((folder / "frozen_policy.json").read_bytes()).hexdigest(),
                      diagnostic={name: run["metrics"] for name, run in evaluation.items()},
                      stress={name: run["metrics"] for name, run in stress.items()},
                      frozen_policy=evaluation[policy["selected_for_execution"]]["metrics"],
                      paired_mse={name: float(np.mean((values - truth) ** 2)) for name, values in prediction.items()},
                      train_mean_mse=float(np.mean((training["train_mean"] - truth) ** 2)))
        result["evaluation_passes"] = passes(evaluation[selected], stress[selected], args.min_trades, config.max_drawdown)
        if selected in ("toric", "ridge"):
            result["evaluation_passes"] &= model_improves(evaluation[selected], evaluation[reference],
                                                         stress[selected], stress[reference])
        folds.append(result)
        write_json(output / "report.json", dict(protocol=protocol, folds=folds, complete=number == len(segments)))
        print(f"Fold {number}: selected={selected}, enabled={enabled}, "
              f"frozen net={result['frozen_policy']['total_return_pct']:.4f}%", flush=True)


if __name__ == "__main__":
    main()
