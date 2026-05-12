#!/usr/bin/env python3
"""Training script for V3 model with Basis and Open Interest."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3
from toric_markov_model.model.trading_model_v3 import ToricTradingModelV3
from toric_markov_model.train import select_device, set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim-angles", type=int, default=64)
    parser.add_argument("--num-states", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--min-pattern-profit", type=float, default=0.003)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--max-pos-weight", type=float, default=15.0)
    parser.add_argument("--pos-weight-mode", type=str, default="sqrt", choices=["ratio", "sqrt", "log"])
    parser.add_argument("--hold-loss-weight", type=float, default=0.2)
    parser.add_argument("--pattern-threshold", type=float, default=0.45)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--binary-focal-gamma", type=float, default=1.0)
    parser.add_argument("--confidence-label-smoothing", type=float, default=0.02)
    parser.add_argument("--aux-loss-weight", type=float, default=0.01)
    parser.add_argument("--non-hold-loss-weight", type=float, default=1.0)
    # Two-stage training
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="0=joint, 1=gate-only, 2=pattern-only")
    parser.add_argument("--resume-from", type=str, default="",
                        help="Optional checkpoint path to warm-start model weights")
    parser.add_argument("--train-encoder-in-stage1", action="store_true",
                        help="If set, stage-1 trains encoder + gate (recommended when training from scratch)")
    parser.add_argument("--gate-epochs", type=int, default=10,
                        help="Epochs for stage-1 (gate-only) training")
    parser.add_argument("--gate-lr", type=float, default=3e-4,
                        help="Learning rate for gate training")
    parser.add_argument("--gate-focal-gamma", type=float, default=2.0,
                        help="Focal gamma for gate (higher = more focus on hard examples)")
    parser.add_argument("--disable-balanced-sampler", action="store_true")
    parser.add_argument("--sampler-pos-scale", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_trading_v3_basis")
    return parser.parse_args()

def build_pos_weight(patterns: torch.Tensor, max_pos_weight: float, mode: str) -> torch.Tensor:
    """Compute class-balancing weights for BCEWithLogits."""
    pos_counts = patterns.sum(dim=0)
    neg_counts = patterns.shape[0] - pos_counts
    ratio = neg_counts / (pos_counts + 1e-6)
    if mode == "ratio":
        pos_weight = ratio
    elif mode == "sqrt":
        pos_weight = torch.sqrt(ratio)
    elif mode == "log":
        pos_weight = torch.log1p(ratio)
    else:
        raise ValueError(f"Unknown pos_weight mode: {mode}")
    pos_weight = torch.clamp(pos_weight, min=1.0, max=max_pos_weight)
    # Last class is hold; keeping it neutral avoids overwhelming all other labels.
    pos_weight[-1] = 1.0
    return pos_weight


def build_class_weight(num_patterns: int, hold_loss_weight: float) -> torch.Tensor:
    class_weight = torch.ones(num_patterns, dtype=torch.float32)
    if num_patterns >= 17:
        class_weight[-1] = hold_loss_weight
    return class_weight


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    class_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    logits = torch.clamp(logits, min=-30.0, max=30.0)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    weighted_bce = bce * class_weight.view(1, -1)
    if gamma <= 0:
        return weighted_bce.mean()
    probs = torch.sigmoid(logits)
    p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
    focal = (1.0 - p_t).pow(gamma)
    return (focal * weighted_bce).mean()


def binary_focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    if gamma <= 0:
        return bce.mean()
    probs = torch.sigmoid(logits)
    p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
    focal = (1.0 - p_t).pow(gamma)
    return (focal * bce).mean()


def update_non_hold_counts(
    predicted_patterns: torch.Tensor,
    target_patterns: torch.Tensor,
    counts: dict[str, float],
) -> None:
    pred = predicted_patterns[:, :-1]
    tgt = target_patterns[:, :-1]
    counts["tp"] += float(((pred == 1) & (tgt == 1)).sum().item())
    counts["fp"] += float(((pred == 1) & (tgt == 0)).sum().item())
    counts["fn"] += float(((pred == 0) & (tgt == 1)).sum().item())


def non_hold_prf(counts: dict[str, float]) -> tuple[float, float, float]:
    precision = counts["tp"] / (counts["tp"] + counts["fp"] + 1e-8)
    recall = counts["tp"] / (counts["tp"] + counts["fn"] + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def update_binary_counts(
    pred_binary: torch.Tensor,
    true_binary: torch.Tensor,
    counts: dict[str, float],
) -> None:
    pred = pred_binary.bool()
    true = true_binary.bool()
    counts["tp"] += float((pred & true).sum().item())
    counts["fp"] += float((pred & ~true).sum().item())
    counts["fn"] += float((~pred & true).sum().item())


def sweep_binary_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    min_thr: float = 0.01,
    max_thr: float = 0.99,
    step: float = 0.01,
) -> tuple[float, float, float, float]:
    best_f1 = -1.0
    best_thr = min_thr
    best_prec = 0.0
    best_rec = 0.0
    thr = min_thr
    while thr <= max_thr + 1e-9:
        pred = scores >= thr
        tp = float((pred & targets).sum())
        fp = float((pred & ~targets).sum())
        fn = float((~pred & targets).sum())
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_prec = prec
            best_rec = rec
        thr += step
    return best_thr, best_prec, best_rec, best_f1


def build_balanced_sampler(
    dataset: TradingDatasetV3,
    pos_scale: float = 1.0,
) -> tuple[WeightedRandomSampler, int, int, float]:
    num_samples = len(dataset)
    aligned_patterns = dataset.patterns[dataset.seq_len:dataset.seq_len + num_samples]
    non_hold = (aligned_patterns[:, :-1].sum(axis=1) > 0).astype(np.float32)
    pos_count = int(non_hold.sum())
    neg_count = int(num_samples - pos_count)
    pos_weight = (neg_count / (pos_count + 1e-6)) * pos_scale
    sample_weights = np.where(non_hold > 0, pos_weight, 1.0).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )
    return sampler, pos_count, neg_count, float(pos_weight)


def configure_stage_trainability(
    model: ToricTradingModelV3,
    stage: int,
    train_encoder_in_stage1: bool,
) -> None:
    """Freeze/unfreeze parameters for requested training stage."""
    for param in model.parameters():
        param.requires_grad = False

    if stage == 0:
        for param in model.parameters():
            param.requires_grad = True
        return

    if stage == 1:
        model.non_hold_gate_head.requires_grad_(True)
        if train_encoder_in_stage1:
            model.feature_emb.requires_grad_(True)
            model.pos_emb.requires_grad_(True)
            model.markov_chain.requires_grad_(True)
            model.toric_layers.requires_grad_(True)
            model.complex_feature_fusion.requires_grad_(True)
            model.feature_anchor.requires_grad_(True)
        return

    if stage == 2:
        model.pattern_head.requires_grad_(True)
        model.confidence_head.requires_grad_(True)
        return

    raise ValueError(f"Unsupported stage: {stage}")


def stage_loss(
    stage: int,
    pattern_loss: torch.Tensor,
    confidence_loss: torch.Tensor,
    non_hold_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    non_hold_loss_weight: float,
    aux_loss_weight: float,
) -> torch.Tensor:
    """Compute total loss according to stage strategy."""
    if stage == 1:
        return non_hold_loss
    if stage == 2:
        return pattern_loss + 0.5 * confidence_loss
    return (
        pattern_loss
        + 0.5 * confidence_loss
        + non_hold_loss_weight * non_hold_loss
        + aux_loss_weight * aux_loss
    )


def stage_primary_metric(stage: int, val_metrics: tuple[float, ...]) -> tuple[str, float]:
    """Return checkpoint-selection metric name and value."""
    if stage == 1:
        return "best_binary_non_hold_f1", val_metrics[20]
    if stage == 2:
        return "non_hold_f1", val_metrics[13]
    return "best_binary_non_hold_f1", val_metrics[20]


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    pos_weight,
    class_weight,
    focal_gamma,
    binary_focal_gamma,
    conf_smoothing,
    aux_loss_weight,
    non_hold_loss_weight,
    non_hold_pos_weight,
    pattern_threshold,
    stage,
):
    model.train()
    total_loss = 0
    total_pattern_loss = 0
    total_confidence_loss = 0
    total_non_hold_loss = 0
    total_return_loss = 0
    total_volume_loss = 0
    total_cvd_loss = 0
    total_poc_loss = 0
    total_basis_loss = 0
    total_oi_loss = 0
    total_correct = 0
    total_samples = 0
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    binary_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    
    for batch_idx, (features, patterns, aux_targets) in enumerate(dataloader):
        features = features.to(device)
        patterns = patterns.to(device)  # [batch, 17] - multi-label
        aux_targets = aux_targets.to(device)  # [batch, 4] - normalized regression targets
        
        outputs = model(features)
        
        non_hold_targets = (patterns[:, :-1].sum(dim=1, keepdim=True) > 0).float()
        non_hold_logits = outputs['non_hold_logit']
        non_hold_loss = binary_focal_bce_with_logits(
            non_hold_logits,
            non_hold_targets,
            pos_weight=non_hold_pos_weight,
            gamma=binary_focal_gamma,
        )

        # Stage-2 pattern loss: optimize pattern type only on positive (non-hold) samples.
        non_hold_mask = non_hold_targets.squeeze(1) > 0.5
        if non_hold_mask.any():
            pattern_loss = focal_bce_with_logits(
                outputs['pattern_logits'][non_hold_mask, :-1],
                patterns[non_hold_mask, :-1],
                pos_weight=pos_weight,
                class_weight=class_weight,
                gamma=focal_gamma,
            )
            confidence_targets = patterns[non_hold_mask, :-1] * (1.0 - 2.0 * conf_smoothing) + conf_smoothing
            confidence_preds = torch.clamp(
                outputs['pattern_confidence'][non_hold_mask, :-1], min=1e-4, max=1.0 - 1e-4
            )
            confidence_loss = F.binary_cross_entropy(confidence_preds, confidence_targets)
        else:
            pattern_loss = torch.zeros((), device=device)
            confidence_loss = torch.zeros((), device=device)
        
        # Confidence head should estimate pattern presence probabilities.
        pattern_probs = torch.sigmoid(outputs['pattern_logits'])
        predicted_patterns = (pattern_probs > pattern_threshold).float()
        pred_binary = (torch.sigmoid(non_hold_logits).squeeze(1) > pattern_threshold)
        true_binary = non_hold_targets.squeeze(1) > 0
        
        # Auxiliary losses on normalized targets.
        return_loss = F.smooth_l1_loss(outputs['predicted_return'], aux_targets[:, 0:1])
        volume_loss = F.smooth_l1_loss(outputs['predicted_volume_change'], aux_targets[:, 1:2])
        cvd_loss = F.smooth_l1_loss(outputs['predicted_cvd_change'], aux_targets[:, 2:3])
        poc_loss = F.smooth_l1_loss(outputs['predicted_poc_movement'], aux_targets[:, 3:4])
        aux_loss = 0.25 * (return_loss + volume_loss + cvd_loss + poc_loss)
        basis_loss = torch.zeros_like(aux_loss)
        oi_loss = torch.zeros_like(aux_loss)

        loss = stage_loss(
            stage=stage,
            pattern_loss=pattern_loss,
            confidence_loss=confidence_loss,
            non_hold_loss=non_hold_loss,
            aux_loss=aux_loss,
            non_hold_loss_weight=non_hold_loss_weight,
            aux_loss_weight=aux_loss_weight,
        )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_pattern_loss += pattern_loss.item()
        total_confidence_loss += confidence_loss.item()
        total_non_hold_loss += non_hold_loss.item()
        total_return_loss += return_loss.item()
        total_volume_loss += volume_loss.item()
        total_cvd_loss += cvd_loss.item()
        total_poc_loss += poc_loss.item()
        total_basis_loss += basis_loss.item()
        total_oi_loss += oi_loss.item()
        
        # Calculate accuracy - pattern detected correctly
        correct = (predicted_patterns == patterns).float().mean()
        total_correct += correct.item() * patterns.size(0)
        total_samples += patterns.size(0)
        update_non_hold_counts(predicted_patterns, patterns, counts)
        update_binary_counts(pred_binary, true_binary, binary_counts)
        
        if (batch_idx + 1) % 10 == 0:
            avg_confidence = outputs['pattern_confidence'].mean().item()
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}, pattern={pattern_loss.item():.4f}, gate={non_hold_loss.item():.4f}, conf={confidence_loss.item():.4f}, avg_conf={avg_confidence:.2f}, acc={100 * correct:.2f}%", flush=True)
    
    n = len(dataloader)
    precision, recall, f1 = non_hold_prf(counts)
    bin_precision, bin_recall, bin_f1 = non_hold_prf(binary_counts)
    return (total_loss/n, total_pattern_loss/n, total_confidence_loss/n, total_non_hold_loss/n, total_return_loss/n, total_volume_loss/n, 
            total_cvd_loss/n, total_poc_loss/n, total_basis_loss/n, total_oi_loss/n,
            100 * total_correct / total_samples, 100 * precision, 100 * recall, 100 * f1,
            100 * bin_precision, 100 * bin_recall, 100 * bin_f1)

def validate(
    model,
    dataloader,
    device,
    pos_weight,
    class_weight,
    focal_gamma,
    binary_focal_gamma,
    conf_smoothing,
    aux_loss_weight,
    non_hold_loss_weight,
    non_hold_pos_weight,
    pattern_threshold,
    stage,
):
    model.eval()
    total_loss = 0
    total_pattern_loss = 0
    total_confidence_loss = 0
    total_non_hold_loss = 0
    total_return_loss = 0
    total_volume_loss = 0
    total_cvd_loss = 0
    total_poc_loss = 0
    total_basis_loss = 0
    total_oi_loss = 0
    total_correct = 0
    total_samples = 0
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    binary_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    all_binary_scores = []
    all_binary_targets = []
    
    with torch.no_grad():
        for features, patterns, aux_targets in dataloader:
            features = features.to(device)
            patterns = patterns.to(device)
            aux_targets = aux_targets.to(device)
            
            outputs = model(features)
            
            non_hold_targets = (patterns[:, :-1].sum(dim=1, keepdim=True) > 0).float()
            non_hold_logits = outputs['non_hold_logit']
            non_hold_loss = binary_focal_bce_with_logits(
                non_hold_logits,
                non_hold_targets,
                pos_weight=non_hold_pos_weight,
                gamma=binary_focal_gamma,
            )

            non_hold_mask = non_hold_targets.squeeze(1) > 0.5
            if non_hold_mask.any():
                pattern_loss = focal_bce_with_logits(
                    outputs['pattern_logits'][non_hold_mask, :-1],
                    patterns[non_hold_mask, :-1],
                    pos_weight=pos_weight,
                    class_weight=class_weight,
                    gamma=focal_gamma,
                )
                confidence_targets = patterns[non_hold_mask, :-1] * (1.0 - 2.0 * conf_smoothing) + conf_smoothing
                confidence_preds = torch.clamp(
                    outputs['pattern_confidence'][non_hold_mask, :-1], min=1e-4, max=1.0 - 1e-4
                )
                confidence_loss = F.binary_cross_entropy(confidence_preds, confidence_targets)
            else:
                pattern_loss = torch.zeros((), device=device)
                confidence_loss = torch.zeros((), device=device)
            
            # Confidence head should estimate pattern presence probabilities.
            pattern_probs = torch.sigmoid(outputs['pattern_logits'])
            predicted_patterns = (pattern_probs > pattern_threshold).float()
            pred_binary = (torch.sigmoid(non_hold_logits).squeeze(1) > pattern_threshold)
            true_binary = (non_hold_targets.squeeze(1) > 0)
            all_binary_scores.append(torch.sigmoid(non_hold_logits).squeeze(1).detach().cpu().numpy())
            all_binary_targets.append(true_binary.detach().cpu().numpy().astype(bool))
            
            return_loss = F.smooth_l1_loss(outputs['predicted_return'], aux_targets[:, 0:1])
            volume_loss = F.smooth_l1_loss(outputs['predicted_volume_change'], aux_targets[:, 1:2])
            cvd_loss = F.smooth_l1_loss(outputs['predicted_cvd_change'], aux_targets[:, 2:3])
            poc_loss = F.smooth_l1_loss(outputs['predicted_poc_movement'], aux_targets[:, 3:4])
            aux_loss = 0.25 * (return_loss + volume_loss + cvd_loss + poc_loss)
            basis_loss = torch.zeros_like(aux_loss)
            oi_loss = torch.zeros_like(aux_loss)
            
            loss = stage_loss(
                stage=stage,
                pattern_loss=pattern_loss,
                confidence_loss=confidence_loss,
                non_hold_loss=non_hold_loss,
                aux_loss=aux_loss,
                non_hold_loss_weight=non_hold_loss_weight,
                aux_loss_weight=aux_loss_weight,
            )
            
            total_loss += loss.item()
            total_pattern_loss += pattern_loss.item()
            total_confidence_loss += confidence_loss.item()
            total_non_hold_loss += non_hold_loss.item()
            total_return_loss += return_loss.item()
            total_volume_loss += volume_loss.item()
            total_cvd_loss += cvd_loss.item()
            total_poc_loss += poc_loss.item()
            total_basis_loss += basis_loss.item()
            total_oi_loss += oi_loss.item()
            
            correct = (predicted_patterns == patterns).float().mean()
            total_correct += correct.item() * patterns.size(0)
            total_samples += patterns.size(0)
            update_non_hold_counts(predicted_patterns, patterns, counts)
            update_binary_counts(pred_binary, true_binary, binary_counts)
    
    n = len(dataloader)
    precision, recall, f1 = non_hold_prf(counts)
    bin_precision, bin_recall, bin_f1 = non_hold_prf(binary_counts)
    all_scores = np.concatenate(all_binary_scores, axis=0)
    all_targets = np.concatenate(all_binary_targets, axis=0).astype(bool)
    best_thr, best_bin_precision, best_bin_recall, best_bin_f1 = sweep_binary_threshold(
        all_scores,
        all_targets,
    )
    return (total_loss/n, total_pattern_loss/n, total_confidence_loss/n, total_non_hold_loss/n, total_return_loss/n, total_volume_loss/n,
            total_cvd_loss/n, total_poc_loss/n, total_basis_loss/n, total_oi_loss/n,
            100 * total_correct / total_samples, 100 * precision, 100 * recall, 100 * f1,
            100 * bin_precision, 100 * bin_recall, 100 * bin_f1,
            float(best_thr), 100 * best_bin_precision, 100 * best_bin_recall, 100 * best_bin_f1)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("=" * 80, flush=True)
    print("V3 MODEL WITH PATTERN DETECTION", flush=True)
    print("Features: Basis, OI, CVD, Volume Profile", flush=True)
    print("Patterns: 17 types (CVD, Basis, OI, Volume Profile)", flush=True)
    print("=" * 80, flush=True)
    print("Loading data...", flush=True)
    
    train_dataset = TradingDatasetV3(
        args.data,
        args.seq_len,
        args.prediction_horizon,
        train=True,
        train_split=args.train_split,
        min_pattern_profit=args.min_pattern_profit,
        return_aux_targets=True,
    )
    norm_stats = train_dataset.get_normalization_stats()
    aux_stats = train_dataset.get_aux_target_stats()
    val_dataset = TradingDatasetV3(
        args.data,
        args.seq_len,
        args.prediction_horizon,
        train=False,
        train_split=args.train_split,
        min_pattern_profit=args.min_pattern_profit,
        normalization_stats=norm_stats,
        aux_target_stats=aux_stats,
        return_aux_targets=True,
    )
    train_oi_ratio = train_dataset.oi_available_ratio
    val_oi_ratio = val_dataset.oi_available_ratio
    train_oi_patterns = float(train_dataset.patterns[:, 12:14].sum())
    if train_oi_ratio < 0.01 and val_oi_ratio > 0.05:
        print(
            f"WARNING: OI mostly in val split (train={train_oi_ratio*100:.2f}%, val={val_oi_ratio*100:.2f}%). "
            "Model cannot learn OI patterns reliably. Consider increasing --train-split or collecting longer OI history.",
            flush=True,
        )
    if train_oi_patterns < 1.0:
        print(
            "WARNING: zero OI pattern labels in train (accumulation/distribution). "
            "OI heads/patterns will be undertrained on current split.",
            flush=True,
        )
    train_sampler = None
    if not args.disable_balanced_sampler:
        train_sampler, pos_count, neg_count, sampler_pos_weight = build_balanced_sampler(
            train_dataset,
            pos_scale=args.sampler_pos_scale,
        )
        print(
            f"Balanced sampler: enabled, pos={pos_count}, neg={neg_count}, "
            f"sample_pos_weight={sampler_pos_weight:.2f}, pos_scale={args.sampler_pos_scale:.2f}",
            flush=True,
        )
    else:
        print("Balanced sampler: disabled", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    num_features = train_dataset.features.shape[1]
    print(f"Features: {num_features}", flush=True)
    pos_weight = build_pos_weight(
        torch.tensor(train_dataset.patterns[:, :-1], dtype=torch.float32),
        max_pos_weight=args.max_pos_weight,
        mode=args.pos_weight_mode,
    ).to(device)
    class_weight = build_class_weight(
        num_patterns=train_dataset.patterns.shape[1] - 1,
        hold_loss_weight=args.hold_loss_weight,
    ).to(device)
    non_hold_targets = (train_dataset.patterns[:, :-1].sum(axis=1) > 0).astype(np.float32)
    non_hold_pos_count = float(non_hold_targets.sum())
    non_hold_neg_count = float(len(non_hold_targets) - non_hold_pos_count)
    non_hold_pos_weight = torch.tensor(
        [np.clip(non_hold_neg_count / (non_hold_pos_count + 1e-6), 1.0, args.max_pos_weight)],
        dtype=torch.float32,
        device=device,
    )
    print(
        f"Pos-weight mode={args.pos_weight_mode}, pattern max={float(pos_weight.max().item()):.2f}, "
        f"pattern mean={float(pos_weight.mean().item()):.2f}, "
        f"binary_non_hold_pos_weight={float(non_hold_pos_weight.item()):.2f}",
        flush=True,
    )
    
    print("Creating V3 PATTERN DETECTION model...", flush=True)
    model = ToricTradingModelV3(
        num_features=num_features,
        dim_angles=args.dim_angles,
        max_len=args.seq_len,
        num_states=args.num_states,
        num_levels=4,
        num_layers=args.num_layers,
        n_bits=8,
        use_attention=True,
        num_patterns=17,  # 16 patterns + hold
        predict_return=True
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    if args.resume_from:
        resume_path = Path(args.resume_from)
        print(f"Loading warm-start checkpoint: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    configure_stage_trainability(
        model=model,
        stage=args.stage,
        train_encoder_in_stage1=args.train_encoder_in_stage1,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    if trainable_count == 0:
        raise RuntimeError("No trainable parameters after stage configuration.")
    current_lr = args.gate_lr if args.stage == 1 else args.lr
    current_binary_focal_gamma = args.gate_focal_gamma if args.stage == 1 else args.binary_focal_gamma
    print(
        f"Stage={args.stage} ({'joint' if args.stage == 0 else 'gate-only' if args.stage == 1 else 'pattern-only'}), "
        f"trainable parameters: {trainable_count:,}, lr={current_lr:.2e}, binary_focal_gamma={current_binary_focal_gamma:.2f}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=current_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    best_val_loss = float('inf')
    best_val_f1 = -1.0
    best_val_threshold = args.pattern_threshold
    patience_counter = 0
    
    total_epochs = args.gate_epochs if args.stage == 1 else args.epochs
    print(f"\nTraining {total_epochs} epochs...", flush=True)
    print("=" * 80, flush=True)
    
    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}", flush=True)
        print("-" * 80, flush=True)
        
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            pos_weight,
            class_weight,
            args.focal_gamma,
            current_binary_focal_gamma,
            args.confidence_label_smoothing,
            args.aux_loss_weight,
            args.non_hold_loss_weight,
            non_hold_pos_weight,
            args.pattern_threshold,
            args.stage,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            pos_weight,
            class_weight,
            args.focal_gamma,
            current_binary_focal_gamma,
            args.confidence_label_smoothing,
            args.aux_loss_weight,
            args.non_hold_loss_weight,
            non_hold_pos_weight,
            args.pattern_threshold,
            args.stage,
        )
        metric_name, metric_value = stage_primary_metric(args.stage, val_metrics)
        scheduler.step(metric_value)
        
        print(f"\nEpoch {epoch + 1} Summary:", flush=True)
        print(
            f"  Train: loss={train_metrics[0]:.4f}, pattern={train_metrics[1]:.4f}, "
            f"gate={train_metrics[3]:.4f}, conf={train_metrics[2]:.4f}, acc={train_metrics[10]:.2f}%, "
            f"non_hold_f1={train_metrics[13]:.2f}% (P={train_metrics[11]:.2f}%, R={train_metrics[12]:.2f}%), "
            f"binary_non_hold_f1@{args.pattern_threshold:.2f}={train_metrics[16]:.2f}%",
            flush=True,
        )
        print(
            f"  Val:   loss={val_metrics[0]:.4f}, pattern={val_metrics[1]:.4f}, "
            f"gate={val_metrics[3]:.4f}, conf={val_metrics[2]:.4f}, acc={val_metrics[10]:.2f}%, "
            f"non_hold_f1={val_metrics[13]:.2f}% (P={val_metrics[11]:.2f}%, R={val_metrics[12]:.2f}%), "
            f"binary_non_hold_f1@{args.pattern_threshold:.2f}={val_metrics[16]:.2f}%, "
            f"best_binary_non_hold_f1={val_metrics[20]:.2f}% at thr={val_metrics[17]:.2f}",
            flush=True,
        )

        improved = (
            metric_value > best_val_f1 + 1e-4
            or (abs(metric_value - best_val_f1) <= 1e-4 and val_metrics[0] < best_val_loss - 1e-4)
        )
        if improved:
            best_val_loss = val_metrics[0]
            best_val_f1 = metric_value
            best_val_threshold = val_metrics[17]
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics[0],
                'val_accuracy': val_metrics[10],
                'val_non_hold_precision': val_metrics[11],
                'val_non_hold_recall': val_metrics[12],
                'val_non_hold_f1': val_metrics[13],
                'val_binary_non_hold_precision_at_pattern_threshold': val_metrics[14],
                'val_binary_non_hold_recall_at_pattern_threshold': val_metrics[15],
                'val_binary_non_hold_f1_at_pattern_threshold': val_metrics[16],
                'best_val_binary_non_hold_threshold': val_metrics[17],
                'best_val_binary_non_hold_precision': val_metrics[18],
                'best_val_binary_non_hold_recall': val_metrics[19],
                'best_val_binary_non_hold_f1': val_metrics[20],
                'primary_metric_name': metric_name,
                'primary_metric_value': metric_value,
                'pos_weight': pos_weight.detach().cpu(),
                'class_weight': class_weight.detach().cpu(),
                'non_hold_pos_weight': non_hold_pos_weight.detach().cpu(),
                'normalization': {
                    'feature_mean': torch.tensor(norm_stats['feature_mean'], dtype=torch.float32),
                    'feature_std': torch.tensor(norm_stats['feature_std'], dtype=torch.float32),
                },
                'aux_target_stats': {
                    'aux_target_mean': torch.tensor(aux_stats['aux_target_mean'], dtype=torch.float32),
                    'aux_target_std': torch.tensor(aux_stats['aux_target_std'], dtype=torch.float32),
                },
                'args': args,
            }, checkpoint_dir / f"best_model_stage{args.stage}.pt")
            if metric_name == "best_binary_non_hold_f1":
                metric_summary = f"best_binary_non_hold_f1={metric_value:.2f}% at thr={val_metrics[17]:.2f}"
            else:
                metric_summary = (
                    f"{metric_name}={metric_value:.2f}%, "
                    f"best_binary_non_hold_f1={val_metrics[20]:.2f}% at thr={val_metrics[17]:.2f}"
                )
            print(
                f"  ✓ Saved best model ({metric_summary}, val_loss={val_metrics[0]:.4f})",
                flush=True,
            )
        else:
            patience_counter += 1
            print(f"  • No improvement ({patience_counter}/{args.early_stop_patience})", flush=True)
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics[0],
                'val_accuracy': val_metrics[10],
                'val_non_hold_precision': val_metrics[11],
                'val_non_hold_recall': val_metrics[12],
                'val_non_hold_f1': val_metrics[13],
                'val_binary_non_hold_precision_at_pattern_threshold': val_metrics[14],
                'val_binary_non_hold_recall_at_pattern_threshold': val_metrics[15],
                'val_binary_non_hold_f1_at_pattern_threshold': val_metrics[16],
                'best_val_binary_non_hold_threshold': val_metrics[17],
                'best_val_binary_non_hold_precision': val_metrics[18],
                'best_val_binary_non_hold_recall': val_metrics[19],
                'best_val_binary_non_hold_f1': val_metrics[20],
                'primary_metric_name': metric_name,
                'primary_metric_value': metric_value,
                'pos_weight': pos_weight.detach().cpu(),
                'class_weight': class_weight.detach().cpu(),
                'non_hold_pos_weight': non_hold_pos_weight.detach().cpu(),
                'normalization': {
                    'feature_mean': torch.tensor(norm_stats['feature_mean'], dtype=torch.float32),
                    'feature_std': torch.tensor(norm_stats['feature_std'], dtype=torch.float32),
                },
                'aux_target_stats': {
                    'aux_target_mean': torch.tensor(aux_stats['aux_target_mean'], dtype=torch.float32),
                    'aux_target_std': torch.tensor(aux_stats['aux_target_std'], dtype=torch.float32),
                },
                'args': args,
            }, checkpoint_dir / f"checkpoint_stage{args.stage}_epoch_{epoch + 1}.pt")

        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}", flush=True)
            break
    
    print(
        f"\nDone! Best val loss: {best_val_loss:.4f}, best stage metric: {best_val_f1:.2f}%, "
        f"best binary non-hold threshold: {best_val_threshold:.2f}",
        flush=True,
    )

if __name__ == "__main__":
    main()
