"""Fit decision thresholds on held-out validation, not probability calibration."""

import numpy as np


def decision_metrics(conditional, gate, labels, thresholds):
    conditional = np.asarray(conditional)
    gate = np.asarray(gate).reshape(-1)
    labels = np.asarray(labels)
    if (conditional.ndim != 2 or conditional.shape[1] < 1 or conditional.shape != labels.shape or
            len(gate) != len(labels) or not len(labels)):
        raise ValueError("nonempty aligned conditional scores, gate and pattern labels required")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("pattern labels must be binary")
    labels = labels.astype(bool)
    if (not np.isfinite(conditional).all() or not np.isfinite(gate).all() or
            ((conditional < 0) | (conditional > 1)).any() or ((gate < 0) | (gate > 1)).any()):
        raise ValueError("scores must be finite and in [0, 1]")
    if any(not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("thresholds must be in [0, 1]")
    active = ((conditional >= thresholds["pattern_prob_threshold"]) &
              (gate[:, None] >= thresholds["gate_threshold"]) &
              (conditional * gate[:, None] >= thresholds["confidence_threshold"]))
    triggered = active.any(axis=1)
    strongest = np.where(active, conditional * gate[:, None], -1).argmax(axis=1)
    events = labels.any(axis=1)
    correct = triggered & labels[np.arange(len(labels)), strongest]
    signals = int(triggered.sum())
    support = int(events.sum())
    hits = int(correct.sum())
    event_hits = int((triggered & events).sum())
    return dict(samples=len(labels), events=support, signals=signals, correct_patterns=hits,
                signal_rate=signals / len(labels), event_prevalence=support / len(labels),
                strongest_precision=hits / max(signals, 1), strongest_recall=hits / max(support, 1),
                strongest_f1=2 * hits / max(signals + support, 1),
                event_precision=event_hits / max(signals, 1), event_recall=event_hits / max(support, 1),
                event_f1=2 * event_hits / max(signals + support, 1))


def fit_thresholds(conditional, gate, labels, min_signals=20, max_signal_rate=0.5):
    if min_signals < 1:
        raise ValueError("min_signals must be positive")
    if not 0 < max_signal_rate < 1:
        raise ValueError("max_signal_rate must be in (0, 1)")
    defaults = dict(gate_threshold=0.5, pattern_prob_threshold=0.5, confidence_threshold=0.0)
    before = decision_metrics(conditional, gate, labels, defaults)
    support = np.asarray(labels, dtype=bool).sum(axis=0)
    constant_hits = int(support.max())
    constant_f1 = 2 * constant_hits / max(before["samples"] + before["events"], 1)
    candidates = []
    for gate_threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for pattern_threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            thresholds = dict(gate_threshold=gate_threshold, pattern_prob_threshold=pattern_threshold,
                              confidence_threshold=0.0)
            metrics = decision_metrics(conditional, gate, labels, thresholds)
            eligible = (metrics["signals"] >= min_signals and
                        metrics["signal_rate"] <= max_signal_rate and
                        metrics["strongest_f1"] > constant_f1 and
                        metrics["event_precision"] > metrics["event_prevalence"])
            candidates.append(dict(thresholds=thresholds, metrics=metrics, eligible=eligible))
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    best = max(eligible, key=lambda candidate: (
        candidate["metrics"]["strongest_f1"], candidate["metrics"]["strongest_precision"],
        -candidate["metrics"]["signals"], candidate["thresholds"]["gate_threshold"],
        candidate["thresholds"]["pattern_prob_threshold"],
    )) if eligible else None
    return dict(accepted=best is not None, objective="strongest_pattern_f1",
                min_signals=min_signals, max_signal_rate=max_signal_rate, default_metrics=before,
                always_hold_f1=0.0, always_most_common_pattern_f1=constant_f1,
                constant_pattern_index=int(support.argmax()),
                decision_thresholds=best["thresholds"] if best else defaults,
                selected_metrics=best["metrics"] if best else before, candidates=candidates)
