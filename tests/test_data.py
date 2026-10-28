"""Data-integrity tests for the raw C-MAPSS files. Run with:  pytest -q tests/test_data.py

These encode the cleaning checks from notebook 01, so a corrupted or wrongly parsed
file is caught before any modelling.
"""

import numpy as np
import pytest

from src.config import RAW_COLS, SENSOR_COLS, SETTING_COLS, USEFUL_SENSORS
from src.data import load_cmapss

EXPECTED = {
    # subset: (train rows, test rows, train engines, test engines)
    "FD001": (20631, 13096, 100, 100),
    "FD002": (53759, 33991, 260, 259),
}


@pytest.fixture(scope="module", params=list(EXPECTED))
def subset(request):
    return request.param, *load_cmapss(request.param)


def test_shapes_and_engine_counts(subset):
    name, train, test, rul = subset
    n_train, n_test, e_train, e_test = EXPECTED[name]
    assert train.shape == (n_train, len(RAW_COLS))
    assert test.shape == (n_test, len(RAW_COLS))
    assert train["unit"].nunique() == e_train
    assert test["unit"].nunique() == e_test
    assert len(rul) == e_test


def test_no_missing_or_infinite_values(subset):
    _, train, test, rul = subset
    for df in (train, test):
        values = df[SETTING_COLS + SENSOR_COLS].to_numpy()
        assert not np.isnan(values).any()
        assert np.isfinite(values).all()
    assert rul.notna().all()


def test_no_duplicate_engine_cycles(subset):
    _, train, test, _ = subset
    for df in (train, test):
        assert not df.duplicated(["unit", "cycle"]).any()


def test_cycles_start_at_one_and_have_no_gaps(subset):
    _, train, test, _ = subset
    for df in (train, test):
        cycles = df.groupby("unit")["cycle"]
        assert (cycles.min() == 1).all()
        assert (cycles.max() == cycles.count()).all()


def test_rul_labels_are_positive_integers(subset):
    _, _, _, rul = subset
    assert (rul > 0).all()
    assert (rul == rul.round()).all()


def test_dropped_fd001_columns_really_are_flat():
    train, _, _ = load_cmapss("FD001")
    for col in ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19", "setting_3"]:
        assert train[col].nunique() == 1, col
    assert train["s_6"].nunique() == 2


def test_kept_sensors_vary():
    train, _, _ = load_cmapss("FD001")
    assert (train[USEFUL_SENSORS].nunique() > 10).all()
