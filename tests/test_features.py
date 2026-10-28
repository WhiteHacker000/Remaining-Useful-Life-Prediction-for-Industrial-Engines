"""Sanity tests for labels and features. Run with:  pytest -q"""

import numpy as np
import pytest

from src.data import add_test_rul, add_train_rul, load_cmapss
from src.features import FeatureBuilder, rolling_slope
from src.models import engine_folds


@pytest.fixture(scope="module")
def fd001():
    train, test, rul = load_cmapss("FD001")
    return add_train_rul(train), test, rul


def test_train_rul_ends_at_zero_and_decreases(fd001):
    train, _, _ = fd001
    assert (train.groupby("unit")["RUL"].min() == 0).all()
    assert (train.groupby("unit")["RUL"].diff().dropna() == -1).all()


def test_test_rul_matches_official_labels(fd001):
    _, test, rul = fd001
    test = add_test_rul(test, rul)
    last = test.loc[test.groupby("unit")["cycle"].idxmax()].set_index("unit")["RUL"]
    assert (last == rul).all()


def test_features_are_causal(fd001):
    """Deleting future cycles must not change any feature of the remaining cycles."""
    train, _, _ = fd001
    fb = FeatureBuilder().fit(train)
    cut = train[train["cycle"] <= 60]
    full = fb.transform(train)
    part = fb.transform(cut)
    cols = fb.feature_names
    np.testing.assert_allclose(full.loc[part.index, cols].to_numpy(), part[cols].to_numpy(), atol=1e-9)


def test_no_missing_features(fd001):
    train, _, _ = fd001
    fb = FeatureBuilder()
    X = fb.fit_transform(train)
    assert not X[fb.feature_names].isna().any().any()


def test_rolling_slope_recovers_a_known_line():
    import pandas as pd
    df = pd.DataFrame({"unit": 1, "cycle": np.arange(1, 41)})
    df["x"] = 3.0 + 0.5 * df["cycle"]
    slope = rolling_slope(df, "x", 10)
    assert np.allclose(slope.iloc[9:], 0.5)


def test_engine_folds_are_disjoint_and_complete():
    folds = engine_folds(range(1, 101), n_folds=5)
    flat = np.concatenate(folds)
    assert len(flat) == len(set(flat)) == 100
