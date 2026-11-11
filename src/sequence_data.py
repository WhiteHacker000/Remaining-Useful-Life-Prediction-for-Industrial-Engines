"""Engine-aware preprocessing and sliding-window datasets for C-MAPSS sequences."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import RUL_CAP, USEFUL_SENSORS


@dataclass(frozen=True)
class SequenceBatch:
    """Fixed-size windows and the endpoint metadata used for evaluation."""

    x: np.ndarray
    y: np.ndarray
    unit: np.ndarray
    cycle: np.ndarray


class EngineStandardizer:
    """Standardize sensor columns using rows from explicitly selected engines."""

    def __init__(self, sensor_columns=USEFUL_SENSORS):
        self.sensor_columns = list(sensor_columns)
        self.scaler_ = None

    def fit(self, frame, training_units):
        units = set(training_units)
        if not units:
            raise ValueError("training_units must contain at least one engine id")
        missing = units.difference(frame["unit"].unique())
        if missing:
            raise ValueError(f"training engine ids are absent from frame: {sorted(missing)}")
        values = frame.loc[frame["unit"].isin(units), self.sensor_columns].to_numpy(
            dtype=np.float64
        )
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("training sensor values must be non-empty and finite")
        self.scaler_ = StandardScaler().fit(values)
        return self

    def transform(self, frame):
        if self.scaler_ is None:
            raise RuntimeError("EngineStandardizer must be fitted before transform")
        values = frame[self.sensor_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("sensor values must be finite")
        transformed = frame.copy()
        normalized = self.scaler_.transform(values)
        for index, column in enumerate(self.sensor_columns):
            transformed[column] = normalized[:, index]
        return transformed


def split_engine_ids(engine_ids, validation_fraction=0.2, seed=42):
    """Return deterministic, disjoint train/validation engine-id arrays."""
    unique = np.unique(np.asarray(list(engine_ids)))
    if len(unique) < 2:
        raise ValueError("at least two distinct engines are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    n_validation = min(
        len(unique) - 1, max(1, int(round(len(unique) * validation_fraction)))
    )
    shuffled = np.random.default_rng(seed).permutation(unique)
    validation = np.sort(shuffled[:n_validation])
    training = np.sort(shuffled[n_validation:])
    return training, validation


def make_sequence_windows(
    frame,
    window_size,
    sensor_columns=USEFUL_SENSORS,
    target_column="RUL",
    target_cap=RUL_CAP,
    stride=1,
    last_only=False,
):
    """Create trailing windows per engine, left-padding short histories by repetition.

    Rows are sorted by engine and cycle before construction. Every target and metadata
    item corresponds to the final timestep of its window. Test-time callers can set
    ``last_only=True`` and ``target_cap=None`` to evaluate uncapped official labels.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")

    sensors = list(sensor_columns)
    required = {"unit", "cycle", target_column, *sensors}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    ordered = frame.sort_values(["unit", "cycle"], kind="stable")
    if ordered.duplicated(["unit", "cycle"]).any():
        raise ValueError("each engine-cycle pair must be unique")

    windows, targets, units, cycles = [], [], [], []
    for unit, engine in ordered.groupby("unit", sort=True):
        cycle_values = engine["cycle"].to_numpy()
        if len(cycle_values) > 1 and not np.all(np.diff(cycle_values) == 1):
            raise ValueError(f"cycles for engine {unit} must be consecutive")

        values = engine[sensors].to_numpy(dtype=np.float32)
        target_values = engine[target_column].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or not np.isfinite(target_values).all():
            raise ValueError("sensor values and targets must be finite")
        if (target_values < 0).any():
            raise ValueError("RUL targets must be non-negative")

        endpoints = [len(engine) - 1] if last_only else range(0, len(engine), stride)
        for end in endpoints:
            start = max(0, end - window_size + 1)
            sequence = values[start : end + 1]
            padding = window_size - len(sequence)
            if padding:
                sequence = np.concatenate(
                    (np.repeat(sequence[:1], padding, axis=0), sequence), axis=0
                )
            target = target_values[end]
            if target_cap is not None:
                target = min(target, target_cap)
            windows.append(sequence)
            targets.append(target)
            units.append(unit)
            cycles.append(cycle_values[end])

    n_features = len(sensors)
    x = np.asarray(windows, dtype=np.float32)
    if not windows:
        x = np.empty((0, window_size, n_features), dtype=np.float32)
    return SequenceBatch(
        x=x,
        y=np.asarray(targets, dtype=np.float32),
        unit=np.asarray(units),
        cycle=np.asarray(cycles),
    )
