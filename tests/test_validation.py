"""Tests for the Phase 2 validation helpers (src/validation.py). Run with:  pytest -q"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from src.dl_models import make_dl_model  # noqa: E402
from src.validation import gradient_saliency, last_rows, paired_wilcoxon  # noqa: E402


def test_last_rows_keeps_the_final_cycle_of_each_engine():
    df = pd.DataFrame({"unit": [2, 2, 1, 1, 1], "cycle": [1, 2, 1, 2, 3], "pred": [5, 4, 9, 8, 7]})
    out = last_rows(df)
    assert out["unit"].tolist() == [1, 2]       # sorted by engine
    assert out["cycle"].tolist() == [3, 2]      # last observed cycle of each engine


def test_paired_wilcoxon_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 125, 100)
    good = y + rng.normal(0, 2, 100)            # small errors
    bad = y + rng.normal(0, 20, 100)            # large errors
    result = paired_wilcoxon(y, good, bad)
    assert result["p-value"] < 0.01
    assert result["mean |error| A"] < result["mean |error| B"]
    assert result["engines where A is better"] > 50


def test_paired_wilcoxon_finds_no_difference_between_equally_good_models():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 125, 100)
    a = y + rng.normal(0, 5, 100)
    b = y + rng.normal(0, 5, 100)               # same error distribution as a
    assert paired_wilcoxon(y, a, b)["p-value"] > 0.05


def test_gradient_saliency_has_one_value_per_time_step_and_channel():
    model = make_dl_model("lstm", n_in=15, window=30)
    x = np.random.default_rng(0).normal(size=(8, 30, 15)).astype(np.float32)
    x[:, :, -1] = 1.0                           # mask channel: all steps real
    saliency = gradient_saliency(model, x)
    assert saliency.shape == (30, 15)
    assert (saliency >= 0).all() and saliency.sum() > 0
