#!/usr/bin/env python3
"""Audit stop distance and compare a frozen grid of rule-based strategies."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from toric_markov_model.data.payoff import walk_forward_segments
from toric_markov_model.execution import ExecutionConfig
from toric_markov_model.strategies import (
    StrategyData, StrategySpec, audit_trade_ledger, calibration_passes, choose_on_selection, evaluate_strategy,
    stop_audit, strategy_grid,
)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--risk-fraction", type=float, default=0.002)
    parser.add_argument("--legacy-trades", help="Optional V3 OPEN/CLOSE ledger; diagnostics only, never used for selection")
    args = parser.parse_args()
    if args.min_trades < 1 or not 0 < args.risk_fraction < 1:
        raise ValueError("positive support and risk fraction in (0, 1) required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("use a new output directory; preserve previous experiments")
    with open(args.data, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    data = StrategyData.from_frame(pd.read_csv(args.data))
    segments = walk_forward_segments(len(data.frame), args.folds)
    grid = {spec.name: spec for spec in strategy_grid()}
    baseline_name = StrategySpec("cvd_rules", "fixed_1pct", 4).name
    protocol = dict(
        version="strategy_stop_v1", data_sha256=digest, args=vars(args),
        execution=asdict(ExecutionConfig()), variants={name: asdict(spec) for name, spec in grid.items()},
        segments=segments, risk_fraction=args.risk_fraction,
        sizing="min(20% capital, risk_fraction / planned stop loss including exit slippage and round-trip fees); gaps can exceed budget",
        atr="14-bar Wilder ATR, SMA seed then recurrence; previous closed bar ATR/close, distances frozen at entry",
        exits="1% or 2% fixed stop with same 2% take; 2 or 3 ATR stop with same 4 ATR take; no trailing",
        signals=dict(
            cvd_rules="same 12 causal CVD/basis conditions as payoff experiment; signed vote",
            trend_breakout="close crosses previous 24-bar high/low, EMA50/200 trend, relative volume24 >1",
            trend_pullback="close recrosses EMA20 in EMA50/200 trend and on correct side of EMA200",
            range_reversion="48-bar close zscore reenters +/-2 band when abs(EMA50-EMA200) <2ATR",
        ),
        selection="48 fixed variants; minimum support; rank by 5th percentile 7-day block-bootstrap daily mean on selection ONLY",
        calibration="only selected variant; min support, positive net return, both calendar halves positive, block lower mean>0, stress return>0; no fallback",
        evaluation="all variants diagnostic; policy frozen before evaluation; year already inspected, not fresh holdout",
        comparisons="same entry opportunity range across horizons; idle equity padded to common segment end; fixed 20% and risk-sized ledgers",
        excluded_initial_history="no learned weights; train prefix provides causal indicator warmup, not policy fitting",
        limitations=["multiple comparisons", "short calibration", "single symbol/year", "hourly intrabar ambiguity",
                     "no funding or borrow", "stop risk is not guaranteed at gaps", "historical inspected data"],
        production_approved=False, live_orders_allowed=False,
    )
    write_json(output / "protocol.json", protocol)
    if args.legacy_trades:
        legacy_path = Path(args.legacy_trades)
        legacy, records = audit_trade_ledger(data, pd.read_csv(legacy_path))
        legacy["ledger_sha256"] = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
        write_json(output / "legacy_stop_audit.json", legacy)
        pd.DataFrame(records).to_csv(output / "legacy_same_entry_counterfactuals.csv", index=False)
    folds, flat_rows = [], []
    for fold_number, segment in enumerate(segments, start=1):
        folder = output / f"fold_{fold_number}"
        folder.mkdir()
        entries = {name: data.entries(*segment[name]) for name in ("selection", "calibration", "evaluation")}
        dates = {name: dict(first_signal_bar=data.frame.timestamp.iloc[indices[0] - 1].isoformat(),
                           first_entry=data.frame.timestamp.iloc[indices[0]].isoformat(),
                           last_entry=data.frame.timestamp.iloc[indices[-1]].isoformat(),
                           evaluation_end=data.frame.timestamp.iloc[indices[-1] + 23].isoformat(),
                           samples=len(indices)) for name, indices in entries.items()}
        print(f"Fold {fold_number} selection: {dates['selection']}", flush=True)
        selection = {name: evaluate_strategy(data, entries["selection"], spec, args.risk_fraction)["metrics"]
                     for name, spec in grid.items()}
        selected = choose_on_selection(selection, args.min_trades)
        policy = dict(selected=selected, enabled=False, min_trades=args.min_trades,
                      selection=selection, calibration=None, calibration_stress=None, dates=dates,
                      production_approved=False, live_orders_allowed=False)
        if selected is not None:
            calibration = evaluate_strategy(data, entries["calibration"], grid[selected], args.risk_fraction)
            stress = evaluate_strategy(data, entries["calibration"], grid[selected], args.risk_fraction, stress=True)
            policy.update(calibration=calibration["metrics"], calibration_stress=stress["metrics"],
                          enabled=calibration_passes(calibration["metrics"], stress["metrics"], args.min_trades))
            pd.DataFrame(calibration["trades"]).to_csv(folder / "calibration_trades.csv", index=False)
        write_json(folder / "frozen_policy.json", policy)
        frozen_digest = hashlib.sha256((folder / "frozen_policy.json").read_bytes()).hexdigest()
        evaluation = {}
        for name, spec in grid.items():
            risk_sized = evaluate_strategy(data, entries["evaluation"], spec, args.risk_fraction)
            fixed_sized = evaluate_strategy(data, entries["evaluation"], spec, risk_fraction=None)
            evaluation[name] = dict(risk_sized=risk_sized["metrics"], fixed_sized=fixed_sized["metrics"])
            evaluation[name]["risk_stress"] = evaluate_strategy(
                data, entries["evaluation"], spec, args.risk_fraction, stress=True)["metrics"]
            evaluation[name]["fixed_stress"] = evaluate_strategy(
                data, entries["evaluation"], spec, risk_fraction=None, stress=True)["metrics"]
            for sizing, run in (("risk_sized", risk_sized), ("fixed_sized", fixed_sized)):
                flat_rows.append(dict(fold=fold_number, name=name, **asdict(spec), sizing=sizing, **run["metrics"]))
                ledger = pd.DataFrame(run["trades"])
                if len(ledger):
                    ledger["entry_time"] = data.frame.timestamp.iloc[ledger.entry.to_numpy()].to_numpy()
                    ledger["exit_time"] = data.frame.timestamp.iloc[ledger.exit.to_numpy()].to_numpy()
                ledger.to_csv(folder / f"{name}__{sizing}_trades.csv", index=False)
        chosen_spec = grid[selected or baseline_name]
        actions = None if policy["enabled"] else np.zeros(len(entries["evaluation"]), dtype=int)
        chosen = evaluate_strategy(data, entries["evaluation"], chosen_spec, args.risk_fraction, actions=actions)
        chosen_stress = evaluate_strategy(data, entries["evaluation"], chosen_spec, args.risk_fraction,
                                          actions=actions, stress=True)
        times = data.frame.timestamp.iloc[entries["evaluation"][0]:entries["evaluation"][-1] + 24]
        pd.DataFrame(dict(timestamp=times.to_numpy(), equity=chosen["equity"])).to_csv(folder / "policy_equity.csv", index=False)
        audit, counterfactuals = stop_audit(data, entries["evaluation"])
        pd.DataFrame(counterfactuals).to_csv(folder / "stopped_trade_counterfactuals.csv", index=False)
        result = dict(fold=fold_number, frozen_policy_sha256=frozen_digest, dates=dates,
                      selected=selected, enabled=policy["enabled"], evaluation=evaluation,
                      policy_metrics=chosen["metrics"], policy_stress=chosen_stress["metrics"], stop_audit=audit)
        folds.append(result)
        write_json(folder / "report.json", result)
        write_json(output / "report.json", dict(protocol=protocol, folds=folds, complete=len(folds) == args.folds))
        pd.DataFrame(flat_rows).to_csv(output / "comparison.csv", index=False)
        print(f"Fold {fold_number}: selected={selected}, enabled={policy['enabled']}; {chosen['metrics']}", flush=True)
    print(f"Finished {output / 'report.json'}; no live orders, no production approval.")


if __name__ == "__main__":
    main()
