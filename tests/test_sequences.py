"""Tests for the Phase 2 window builder. Run with:  pytest -q"""

import numpy as np
from src.data import add_train_rul, load_cmapss
from src.sequences import SequenceBuilder, make_windows



def test_windows_never_cross_engines():
    # two engines with clearly different values
    values = np.r_[np.full((5, 1), 1.0), np.full((4, 1), 2.0)]
    units = np.r_[np.full(5, 1), np.full(4, 2)]
    X = make_windows(values, units, window=6)
    assert np.all(X[:5, :, 0] == 1.0)          # engine 1 windows only contain engine 1
    assert np.all(X[5:, :, 0] == 2.0)          # engine 2 windows only contain engine 2


def test_window_ends_at_current_cycle_and_mask_marks_padding():
    values = np.arange(10, dtype=float).reshape(-1, 1)
    X = make_windows(values, np.ones(10), window=4)
    assert X[6, -1, 0] == 6                      # last step = current cycle
    assert list(X[6, :, 0]) == [3, 4, 5, 6]      # only past cycles
    assert list(X[1, :, 0]) == [0, 0, 0, 1]      # left-padded with the first reading
    assert list(X[1, :, 1]) == [0, 0, 1, 1]      # mask: 0 = padding, 1 = real


def test_windows_are_causal_on_real_data():
    train, _, _ = load_cmapss("FD001")
    train = add_train_rul(train)
    sb = SequenceBuilder(window=30).fit(train)
    full = sb.transform(train)
    cut = sb.transform(train[train["cycle"] <= 50])
    rows = full["cycle"] <= 50          # both are sorted by engine, then cycle
    np.testing.assert_allclose(full["x"][rows], cut["x"], atol=1e-6)
    np.testing.assert_allclose(full["f"][rows], cut["f"], atol=1e-5)
