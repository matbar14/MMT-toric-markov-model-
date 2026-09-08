"""Shared supervised losses and train/validation execution."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def build_pos_weight(patterns: torch.Tensor, max_pos_weight: float = 5.0,
                     mode: str = "sqrt") -> torch.Tensor:
    if patterns.ndim != 2 or patterns.shape[0] == 0 or max_pos_weight < 1:
        raise ValueError("nonempty label matrix and max_pos_weight >= 1 required")
    positive = patterns.sum(dim=0)
    ratio = (patterns.shape[0] - positive) / positive.clamp_min(1)
    if mode == "sqrt":
        ratio = ratio.sqrt()
    elif mode == "log":
        ratio = ratio.log1p()
    elif mode != "ratio":
        raise ValueError("unknown positive-weight mode")
    return torch.where(positive > 0, ratio.clamp(1, max_pos_weight), torch.ones_like(ratio))


def configure_stage_trainability(model, stage: int, train_encoder_in_stage1: bool = True):
    if stage not in (0, 1, 2):
        raise ValueError("stage must be 0, 1 or 2")
    model.requires_grad_(stage == 0)
    if stage == 1:
        if train_encoder_in_stage1:
            model.requires_grad_(True)
            model.pattern_head.requires_grad_(False)
            if model.predict_return:
                model.aux_head.requires_grad_(False)
        model.non_hold_gate_head.requires_grad_(True)
    if stage == 2:
        model.pattern_head.requires_grad_(True)


def compute_loss(model, outputs, patterns, auxiliary, pos_weight, gate_weight,
                 stage: int = 0, aux_loss_weight: float = 0.01):
    targets = patterns[:, :-1]
    event = targets.any(dim=1)
    if event.any():
        pattern_loss = functional.binary_cross_entropy_with_logits(
            outputs["pattern_logits"][event, :-1], targets[event], pos_weight=pos_weight,
        )
    else:
        pattern_loss = outputs["pattern_logits"][:, :-1].sum() * 0
    gate_loss = functional.binary_cross_entropy_with_logits(
        outputs["non_hold_logit"], event[:, None].float(), pos_weight=gate_weight,
    )
    aux_loss = pattern_loss * 0
    if stage == 0 and model.predict_return:
        prediction = torch.cat([outputs[name] for name in model.AUX_NAMES], dim=1)
        aux_loss = functional.smooth_l1_loss(prediction, auxiliary)
    if stage == 1:
        total = gate_loss
    elif stage == 2:
        total = pattern_loss
    elif stage == 0:
        total = pattern_loss + gate_loss + aux_loss_weight * aux_loss
    else:
        raise ValueError("unknown stage")
    return total, {"pattern_loss": pattern_loss, "gate_loss": gate_loss, "aux_loss": aux_loss}


def run_epoch(model, dataloader, device, pos_weight, gate_weight, stage=0,
              aux_loss_weight=0.01, optimizer=None, thresholds=None):
    training = optimizer is not None
    model.train(training)
    if training and stage == 2:
        model.eval()
        model.pattern_head.train()
    thresholds = thresholds or {}
    totals = dict(loss=0.0, pattern_loss=0.0, gate_loss=0.0, aux_loss=0.0)
    pattern_counts = torch.zeros(3, model.num_patterns - 1, dtype=torch.float64)
    gate_counts = torch.zeros(3, dtype=torch.float64)
    samples = 0
    event_samples = 0
    updates = 0
    with torch.set_grad_enabled(training):
        for features, patterns, auxiliary in dataloader:
            features, patterns, auxiliary = (value.to(device) for value in (features, patterns, auxiliary))
            outputs = model(features)
            loss, components = compute_loss(model, outputs, patterns, auxiliary,
                                            pos_weight, gate_weight, stage, aux_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss; checkpoint not saved")
            event_count = int(patterns[:, :-1].any(1).sum().item())
            event_samples += event_count
            if training:
                optimizer.zero_grad(set_to_none=True)
                if stage != 2 or event_count:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
            batch_size = features.shape[0]
            samples += batch_size
            for name, value in components.items():
                count = event_count if name == "pattern_loss" else batch_size
                totals[name] += value.item() * count
            with torch.no_grad():
                decision = model.decode_outputs(outputs, **thresholds)
                predicted = decision["active_patterns"][:, :-1]
                truth = patterns[:, :-1].bool()
                counts = torch.stack(((predicted & truth).sum(0), (predicted & ~truth).sum(0),
                                      (~predicted & truth).sum(0)))
                pattern_counts += counts.cpu()
                gate_prediction = decision["non_hold_prob"] >= thresholds.get("gate_threshold", 0.5)
                gate_truth = truth.any(1)
                gate_counts += torch.stack(((gate_prediction & gate_truth).sum(),
                                             (gate_prediction & ~gate_truth).sum(),
                                             (~gate_prediction & gate_truth).sum())).cpu()
    if samples == 0:
        raise ValueError("empty dataloader")
    result = {name: value / samples for name, value in totals.items()}
    result["pattern_loss"] = totals["pattern_loss"] / max(1, event_samples)
    if stage == 0:
        result["loss"] = result["pattern_loss"] + result["gate_loss"] + aux_loss_weight * result["aux_loss"]
    else:
        result["loss"] = result["gate_loss"] if stage == 1 else result["pattern_loss"]
    for prefix, counts in (("pattern", pattern_counts.sum(1)), ("gate", gate_counts)):
        true_positive, false_positive, false_negative = counts.tolist()
        result[f"{prefix}_precision"] = true_positive / max(1, true_positive + false_positive)
        result[f"{prefix}_recall"] = true_positive / max(1, true_positive + false_negative)
        result[f"{prefix}_f1"] = 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)
    result["per_pattern_counts"] = pattern_counts.tolist()
    result["samples"] = samples
    result["optimizer_updates"] = updates
    return result
