"""Metrics and error breakdowns for RUL prediction."""

import numpy as np
import pandas as pd

from .config import SEED

# Degradation stages used for the error breakdown (true RUL, in cycles).
STAGES = {
    "early (RUL > 125)": (125, np.inf),
    "mid (50 < RUL <= 125)": (50, 125),
    "late (RUL <= 50)": (-np.inf, 50),
}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def nasa_score(y_true, y_pred):
    """Asymmetric scoring function from the PHM08 challenge (Saxena et al., 2008).

    d = predicted - true. Late predictions (d > 0) are penalised more than early ones,
    because over-estimating the remaining life can mean flying an engine past failure.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def summarize(y_true, y_pred):
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "Score": nasa_score(y_true, y_pred)}


def stage_of(rul):
    rul = np.asarray(rul)
    labels = np.empty(len(rul), dtype=object)
    for name, (lo, hi) in STAGES.items():
        labels[(rul > lo) & (rul <= hi)] = name
    return labels


def errors_by_stage(y_true, y_pred, stage_rul=None):
    """Error per degradation stage. Stages come from `stage_rul` (defaults to y_true),
    so predictions can be scored against a capped target while staging by true RUL."""
    stage_rul = y_true if stage_rul is None else stage_rul
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "stage": stage_of(stage_rul)})
    rows = []
    for name in STAGES:
        part = df[df["stage"] == name]
        rows.append({"stage": name, "n": len(part),
                     "RMSE": rmse(part["true"], part["pred"]),
                     "MAE": mae(part["true"], part["pred"]),
                     "mean error (pred - true)": float((part["pred"] - part["true"]).mean())})
    return pd.DataFrame(rows).set_index("stage")


def bootstrap_rmse(y_true, y_pred, n_boot=2000, seed=SEED):
    """95% bootstrap interval for RMSE, resampling engines with replacement."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    boots = np.sqrt(np.mean((y_pred[idx] - y_true[idx]) ** 2, axis=1))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
