"""Validation helpers for the final test evaluation (Phase 2)."""

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


def last_rows(df):
    """The last observed cycle of every engine, sorted by engine id."""
    return df.loc[df.groupby("unit")["cycle"].idxmax()].sort_values("unit").reset_index(drop=True)


def paired_wilcoxon(y_true, pred_a, pred_b):
    """Wilcoxon signed-rank test on per-engine absolute errors of two models.

    Paired: both models are scored on the same engines, so we test whether the
    median of |e_a| - |e_b| is zero. Makes no normality assumption.
    """
    ea, eb = np.abs(np.asarray(pred_a) - y_true), np.abs(np.asarray(pred_b) - y_true)
    stat, p = wilcoxon(ea, eb)
    return {"mean |error| A": ea.mean(), "mean |error| B": eb.mean(),
            "engines where A is better": int((ea < eb).sum()), "statistic": stat, "p-value": p}


def gradient_saliency(model, x, f=None):
    """Mean |d prediction / d input| over a batch of windows -> (time steps, channels).

    Shows which sensors and which part of the window the prediction is most sensitive to.
    """
    model.eval()
    x = torch.as_tensor(x).clone().requires_grad_(True)
    f = None if f is None else torch.as_tensor(f)
    model(x, f).sum().backward()
    return x.grad.abs().mean(dim=0).numpy()
