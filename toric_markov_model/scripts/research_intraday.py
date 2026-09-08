#!/usr/bin/env python3
"""Frozen intraday spot baseline screening, then conditional Toric/Ridge comparison."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from toric_markov_model.data.payoff import walk_forward_segments
from toric_markov_model.signal_filter import (
    ENTRY_HOURS_UTC, SIGNALS, SignalData, base_screen, filter_actions,
    matched_random_control, paired_comparison, predict_filters, run_signal,
    select_signal, train_entry_filters,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-trades", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1 or args.min_trades < 1:
        raise ValueError("positive epochs and minimum trades required")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory")
    torch.set_num_threads(1)
    data = SignalData.from_frame(pd.read_csv(args.data))
    segments = walk_forward_segments(len(data.frame), args.folds)
    seq_len = 32
    all_entries = [{name: data.entries(*bounds, seq_len) for name, bounds in segment.items()}
                   for segment in segments]
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), rows=len(data.frame),
        start_inclusive=data.frame.timestamp.iloc[0].isoformat(),
        end_exclusive=(data.frame.timestamp.iloc[-1] + pd.Timedelta(hours=1)).isoformat(),
        signals=list(SIGNALS), entry_hours_utc=list(ENTRY_HOURS_UTC), execution=asdict(data.config),
        intraday="close no later than end of six-hour slot; no overnight UTC positions",
        decisions="previous closed candle only; buy at next open; long/flat spot, no leverage",
        selection="rank fixed three signals on selection; require independent calibration pass before fitting filters",
        base_screen="at least min_trades, 0.5 trades/day, positive net/stress return and block lower daily mean",
        filter_screen="paired daily improvement, net/stress gain, no worse drawdown; retain >=50% and >=0.5 trades/day",
        min_trades=args.min_trades, seq_len=seq_len, epochs=args.epochs, seeds=[42, 43, 44],
        segments=segments, stress="fee x1.5 and slippage x2 per side", production_approved=False,
        live_orders_allowed=False,
        limitations=["development data, not independent holdout", "multiple comparisons, not proof of significance",
                     "one symbol", "fixed 2% stop not demonstrated optimal", "hourly intrabar stop-first approximation"],
    )
    write_json(output / "protocol.json", protocol)
    folds = []
    for number, entries in enumerate(all_entries, 1):
        folder = output / f"fold_{number}"
        folder.mkdir()
        selection = {name: run_signal(data, entries["selection"], data.actions(name, entries["selection"]))["metrics"]
                     for name in SIGNALS}
        selected = select_signal(selection, args.min_trades)
        policy = dict(selected=selected, selection=selection, base_enabled=False, filter_enabled={},
                      calibration=None, calibration_filters={}, histories=[], live_orders_allowed=False)
        models = ridge = None
        if selected is not None:
            actions = data.actions(selected, entries["calibration"])
            base = run_signal(data, entries["calibration"], actions)
            stress = run_signal(data, entries["calibration"], actions, stress=True)
            policy["calibration"] = dict(base=base["metrics"], stress=stress["metrics"])
            policy["base_enabled"] = base_screen(base, stress, args.min_trades)
            train_entries = entries["train"][data.actions(selected, entries["train"]).astype(bool)]
            selection_entries = entries["selection"][data.actions(selected, entries["selection"]).astype(bool)]
            if policy["base_enabled"] and len(train_entries) >= 20 and len(selection_entries):
                models, ridge, policy["histories"] = train_entry_filters(
                    data, train_entries, selection_entries, seq_len, args.epochs)
                torch.save(dict(format="toric_intraday_filter_v1", configs=[model.config for model in models],
                                states=[model.state_dict() for model in models]), folder / "filter_weights.pt")
                normalizer, estimator = ridge.steps[0][1], ridge.steps[1][1]
                np.savez(folder / "ridge_weights.npz", mean=normalizer.mean_, scale=normalizer.scale_,
                         coef=estimator.coef_, intercept=estimator.intercept_)
                policy["weights_sha256"] = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                            for name in ("filter_weights.pt", "ridge_weights.npz")}
                forecasts, _ = predict_filters(models, ridge, data.windows(entries["calibration"], seq_len))
                for name, values in forecasts.items():
                    filtered_actions = filter_actions(actions, values)
                    filtered = run_signal(data, entries["calibration"], filtered_actions)
                    filtered_stress = run_signal(data, entries["calibration"], filtered_actions, stress=True)
                    comparison = paired_comparison(base, filtered, stress, filtered_stress, args.min_trades)
                    policy["calibration_filters"][name] = comparison
                    policy["filter_enabled"][name] = comparison["improved"]
            else:
                policy["filter_skip_reason"] = "baseline rejected or insufficient prior fitting entries"
        frozen = folder / "frozen_policy.json"
        write_json(frozen, policy)
        evaluation_entries = entries["evaluation"]
        actions = (data.actions(selected, evaluation_entries) if selected is not None
                   else np.zeros(len(evaluation_entries), dtype=int))
        base = run_signal(data, evaluation_entries, actions)
        stress = run_signal(data, evaluation_entries, actions, stress=True)
        deployed = run_signal(data, evaluation_entries, actions if policy["base_enabled"] else np.zeros_like(actions))
        result = dict(fold=number, selected=selected, base_enabled=policy["base_enabled"],
                      frozen_policy_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                      baseline_diagnostic=base["metrics"], baseline_stress=stress["metrics"],
                      baseline_passes_evaluation=base_screen(base, stress, args.min_trades),
                      frozen_base_policy=deployed["metrics"], filters={})
        pd.DataFrame(base["trades"]).to_csv(folder / "baseline_trades.csv", index=False)
        dates = data.frame.timestamp.iloc[evaluation_entries[0]:evaluation_entries[-1] + data.config.horizon]
        pd.DataFrame(dict(timestamp=dates, baseline=base["equity"], frozen_base=deployed["equity"])).to_csv(
            folder / "equity.csv", index=False)
        if models is not None:
            forecasts, members = predict_filters(models, ridge, data.windows(evaluation_entries, seq_len))
            np.savez(folder / "evaluation_forecasts.npz", entries=evaluation_entries,
                     toric_members=np.asarray(members), **forecasts)
            for name, values in forecasts.items():
                filtered_actions = filter_actions(actions, values)
                filtered = run_signal(data, evaluation_entries, filtered_actions)
                filtered_stress = run_signal(data, evaluation_entries, filtered_actions, stress=True)
                result["filters"][name] = dict(
                    enabled=policy["filter_enabled"][name], diagnostic=filtered["metrics"],
                    stress=filtered_stress["metrics"],
                    comparison=paired_comparison(base, filtered, stress, filtered_stress, args.min_trades),
                    random_control=matched_random_control(data, evaluation_entries, actions, filtered_actions),
                    frozen_policy=(filtered if policy["filter_enabled"][name] else base)["metrics"],
                )
                pd.DataFrame(filtered["trades"]).to_csv(folder / f"{name}_diagnostic_trades.csv", index=False)
        folds.append(result)
        write_json(output / "report.json", dict(protocol=protocol, folds=folds, complete=number == len(all_entries)))
        print(f"Fold {number}: {selected}, calibration_pass={policy['base_enabled']}, "
              f"evaluation trades={base['metrics']['trades']}, "
              f"net={base['metrics']['total_return_pct']:.3f}%, "
              f"stress={stress['metrics']['total_return_pct']:.3f}%", flush=True)


if __name__ == "__main__":
    main()
