"""Restricted experimental checkpoint loading and closed-bar paper inference."""

import numpy as np
import pandas as pd
import torch

from ..data.payoff import causal_context, validate_market
from ..execution import ExecutionConfig
from ..model.payoff import ToricPayoffModel
from .payoff import choose_actions


def load_payoff_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "toric_payoff_v1":
        raise ValueError("unsupported payoff checkpoint format; do not load legacy classifiers here")
    config = ExecutionConfig(**checkpoint["execution"])
    if checkpoint["toric_policy"]["execution"] != checkpoint["execution"]:
        raise ValueError("policy costs and model target costs differ")
    model = ToricPayoffModel(**checkpoint["model_config"])
    state = checkpoint["model_state_dict"]
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite checkpoint tensors")
    model.load_state_dict(state, strict=True)
    if not model.fitted_statistics.item() or (model.feature_std <= 0).any() or (model.target_std <= 0).any():
        raise ValueError("missing or invalid train statistics")
    if len(checkpoint["feature_names"]) != model.num_features:
        raise ValueError("feature schema mismatch")
    policy = checkpoint["toric_policy"]
    if (not isinstance(policy["enabled"], bool) or not isinstance(policy["pattern_filter"], bool) or
            not np.isfinite(policy["threshold"]) or policy["threshold"] < 0):
        raise ValueError("invalid payoff policy")
    return checkpoint, model.eval(), config


def forecast_closed_bars(checkpoint, model, frame):
    """A paper-only next-open proposal. Caller must supply completed candles only."""
    frame = validate_market(frame)
    if len(frame) < 100 + model.max_len:
        raise ValueError("insufficient completed history for indicators and sequence")
    features, names, patterns = causal_context(frame)
    if names != checkpoint["feature_names"]:
        raise ValueError("market feature schema differs from checkpoint")
    interval = frame.timestamp.iloc[-1] - frame.timestamp.iloc[-2]
    training_dates = checkpoint["dates"]["train"]
    trained_interval = (pd.Timestamp(training_dates["first_entry"]) -
                        pd.Timestamp(training_dates["first_context"])) / model.max_len
    if interval != trained_interval:
        raise ValueError("bar interval differs from the training data")
    next_entry = frame.timestamp.iloc[-1] + interval
    if next_entry <= pd.Timestamp(checkpoint["dates"]["calibration"]["last_target"]):
        raise ValueError("forecast must be after the model calibration period")
    scores = model.predict_payoffs(torch.from_numpy(features[-model.max_len:][None])).numpy()
    policy = checkpoint["toric_policy"]
    enabled = policy["enabled"] and checkpoint["selected_model"] == "toric"
    allowed = np.array([[patterns[-1, ::2].any(), patterns[-1, 1::2].any()]]) if policy["pattern_filter"] else None
    action = int(choose_actions(scores, policy["threshold"], allowed)[0]) if enabled else 0
    return dict(next_entry=next_entry.isoformat(), expected_net_long=float(scores[0, 0]),
                expected_net_short=float(scores[0, 1]), paper_action={0: "HOLD", 1: "LONG", -1: "SHORT"}[action],
                policy_enabled=bool(enabled), architecture_selected=checkpoint["selected_model"],
                production_approved=False, live_orders_allowed=False,
                reason="experimental paper policy" if enabled else "model selection or calibration rejected trading",
                assumptions="completed candles only; next-open execution at saved costs; no funding/borrow")
