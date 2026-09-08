#!/usr/bin/env python3
"""Predeclared expanding-window payoff experiment; no live trading or profit guarantee."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from toric_markov_model.data.payoff import PATTERN_NAMES, PayoffData, walk_forward_segments
from toric_markov_model.execution import ExecutionConfig
from toric_markov_model.train.checkpoint import save_checkpoint
from toric_markov_model.train.payoff import (
    choose_actions, fit_baselines, policy_actions, predict_all, select_policy, simulate, train_toric,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pattern_audit(data, entries):
    patterns = data.candidates[entries - 1]
    terminal = data.frame.spot_close.to_numpy()[entries + data.config.horizon - 1] / data.frame.spot_open.to_numpy()[entries] - 1
    report = {}
    for index, name in enumerate(PATTERN_NAMES):
        selected = patterns[:, index]
        column, side = index % 2, 1 if index % 2 == 0 else -1
        payoffs = data.net_returns[entries[selected], column]
        old_positive = selected & (side * terminal > 0.003)
        report[name] = dict(candidates=int(selected.sum()),
                            mean_net_return=float(payoffs.mean()) if len(payoffs) else None,
                            profitable_fraction=float((payoffs > 0).mean()) if len(payoffs) else None,
                            old_positive_labels=int(old_positive.sum()),
                            old_positive_but_net_loss=int((old_positive & (data.net_returns[entries, column] <= 0)).sum()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.seq_len, args.min_trades) < 1:
        raise ValueError("positive epochs, context and support required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("use a new output directory for every experiment")
    with open(args.data, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    torch.set_num_threads(1)
    config = ExecutionConfig()
    data = PayoffData.from_frame(pd.read_csv(args.data), config)
    segments = walk_forward_segments(len(data.frame), args.folds)
    protocol = dict(
        version="payoff_research_v1", data_sha256=digest, args=vars(args), execution=asdict(config),
        feature_names=data.feature_names, pattern_names=PATTERN_NAMES, folds=segments,
        architecture_selection="minimum unscaled selection MSE versus train-mean, ridge and tree baselines",
        target="expected net long/short return at next-open entry, stop-first exit, fees and adverse slippage",
        policy_selection="calibration only; 5 net-edge thresholds x with/without causal pattern filter",
        screening="20 trades by default, <=50% signals, both calendar halves positive, 7-day block lower mean >0",
        evaluation="historical developmental walk-forward; year already inspected, not a fresh holdout",
        feature_policy="past-only stationary features; CVD differences, not arbitrary cumulative origin",
        exclusions="OI absent; POC approximation removed from new policy; legacy v3 remains unchanged",
        limitations=["single market/year", "overlapping outcome labels", "no funding/borrow",
                     "calibration selection uncertainty", "no live execution"],
        production_approved=False,
    )
    write_json(output / "protocol.json", protocol)
    results = []
    for fold, segment in enumerate(segments, start=1):
        folder = output / f"fold_{fold}"
        folder.mkdir()
        entries = {name: data.entries(start, end, args.seq_len) for name, (start, end) in segment.items()}
        dates = {name: dict(first_context=data.frame.timestamp.iloc[indices[0] - args.seq_len].isoformat(),
                           first_entry=data.frame.timestamp.iloc[indices[0]].isoformat(),
                           last_target=data.frame.timestamp.iloc[indices[-1] + config.horizon - 1].isoformat(),
                           samples=len(indices)) for name, indices in entries.items()}
        print(f"Fold {fold}: {dates}", flush=True)
        model, training = train_toric(data, entries, seq_len=args.seq_len, epochs=args.epochs, seed=args.seed)
        baseline = fit_baselines(data.windows(entries["train"], args.seq_len), data.net_returns[entries["train"]])
        selection_scores = predict_all(model, baseline, data.windows(entries["selection"], args.seq_len))
        mse = {name: float(np.square(scores - data.net_returns[entries["selection"]]).mean())
               for name, scores in selection_scores.items()}
        selected_model = min(mse, key=mse.get)
        calibration_scores = predict_all(model, baseline, data.windows(entries["calibration"], args.seq_len))
        policies = {name: select_policy(data, entries["calibration"], scores, args.min_trades)
                    for name, scores in calibration_scores.items()}
        write_json(folder / "selection.json", dict(mse=mse, selected_model=selected_model, dates=dates))
        write_json(folder / "policies.json", policies)
        save_checkpoint(dict(format="toric_payoff_v1", model_config=model.config,
                             model_state_dict=model.state_dict(), feature_names=data.feature_names,
                             data_sha256=digest, execution=asdict(config), training=training,
                             toric_policy=policies["toric"], selected_model=selected_model,
                             dates=dates, production_approved=False), folder / "toric_payoff.pt")
        joblib.dump(baseline, folder / "baselines.joblib")
        evaluation_entries = entries["evaluation"]
        evaluation_scores = predict_all(model, baseline, data.windows(evaluation_entries, args.seq_len))
        evaluation = {}
        prediction_table = pd.DataFrame(dict(timestamp=data.frame.timestamp.iloc[evaluation_entries].to_numpy(),
                                             net_long=data.net_returns[evaluation_entries, 0],
                                             net_short=data.net_returns[evaluation_entries, 1]))
        for name, scores in evaluation_scores.items():
            variants = dict(
                fixed_zero_edge=choose_actions(scores, 0),
                fixed_zero_edge_patterns=choose_actions(scores, 0, data.eligible_sides(evaluation_entries)),
                screened=policy_actions(data, evaluation_entries, scores, policies[name]),
            )
            evaluation[name] = dict(mse=float(np.square(scores - data.net_returns[evaluation_entries]).mean()), variants={})
            prediction_table[name + "_long"] = scores[:, 0]
            prediction_table[name + "_short"] = scores[:, 1]
            for variant, actions in variants.items():
                run = simulate(data, evaluation_entries, actions)
                evaluation[name]["variants"][variant] = run["metrics"]
                if variant == "screened":
                    stress = replace(config, fee=config.fee * 1.5, slippage=config.slippage * 2)
                    evaluation[name]["cost_stress"] = simulate(data, evaluation_entries, actions, stress)["metrics"]
                pd.DataFrame(run["trades"]).to_csv(folder / f"{name}_{variant}_trades.csv", index=False)
                prediction_table[name + "_" + variant + "_action"] = actions
        patterns = data.candidates[evaluation_entries - 1]
        rule_actions = np.sign(patterns[:, ::2].sum(1) - patterns[:, 1::2].sum(1))
        rules = simulate(data, evaluation_entries, rule_actions)
        pd.DataFrame(rules["trades"]).to_csv(folder / "rules_only_trades.csv", index=False)
        prediction_table.to_csv(folder / "evaluation_predictions.csv", index=False)
        result = dict(fold=fold, dates=dates, training=training, selection_mse=mse, selected_model=selected_model,
                      policy_enabled=policies[selected_model]["enabled"], evaluation=evaluation,
                      selected_policy=evaluation[selected_model]["variants"]["screened"],
                      rules_only=rules["metrics"], pattern_audit=pattern_audit(data, evaluation_entries))
        results.append(result)
        write_json(folder / "report.json", result)
        write_json(output / "report.json", dict(protocol=protocol, folds=results, complete=len(results) == args.folds))
        print(f"Fold {fold} selected {selected_model}; policy enabled={result['policy_enabled']}; "
              f"metrics={result['selected_policy']}", flush=True)
    print(f"Finished: {output / 'report.json'}. No production approval or live orders.")


if __name__ == "__main__":
    main()
