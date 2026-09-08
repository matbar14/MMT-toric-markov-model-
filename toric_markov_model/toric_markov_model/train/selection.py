"""Chronologically separate epoch selection from decision-threshold fitting."""

from copy import copy
import math


def partition_validation(dataset, calibration_fraction=0.5):
    if dataset.split != "validation":
        raise ValueError("only validation may be partitioned for selection/calibration")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration fraction must be in (0, 1)")
    boundary = int(len(dataset.features) * (1 - calibration_fraction))
    partitions = []
    for selection in (slice(None, boundary), slice(boundary, None)):
        part = copy(dataset)
        for name in ("features", "patterns", "aux_targets", "timestamps", "spot_open", "spot_high",
                     "spot_low", "spot_close"):
            values = getattr(dataset, name)[selection]
            if hasattr(values, "reset_index"):
                values = values.reset_index(drop=True)
            setattr(part, name, values)
        if len(part) < 1:
            raise ValueError("validation is too short to separate selection and calibration windows")
        partitions.append(part)
    metadata = dict(
        method="chronological_disjoint_windows", calibration_fraction=calibration_fraction,
        selection_start=partitions[0].timestamps.iloc[0].isoformat(),
        selection_end=partitions[0].timestamps.iloc[-1].isoformat(),
        calibration_start=partitions[1].timestamps.iloc[0].isoformat(),
        calibration_end=partitions[1].timestamps.iloc[-1].isoformat(),
        selection_samples=len(partitions[0]), calibration_samples=len(partitions[1]),
    )
    return *partitions, metadata


def improves_loss(value, best, min_delta=1e-4):
    if not math.isfinite(value):
        raise ValueError("selection loss must be finite")
    return value < best - min_delta
