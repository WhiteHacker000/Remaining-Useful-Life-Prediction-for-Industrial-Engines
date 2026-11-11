"""Tests for engine-isolated sequence preprocessing and RUL alignment."""

import numpy as np
import pandas as pd
import pytest

from src.sequence_data import (
    EngineStandardizer,
    make_sequence_windows,
    split_engine_ids,
)


@pytest.fixture
def toy_engines():
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "sensor": [1.0, 2.0, 3.0, 100.0, 101.0],
            "RUL": [10, 9, 8, 130, 129],
        }
    )


def test_engine_split_is_disjoint_complete_and_reproducible():
    first_train, first_validation = split_engine_ids(range(1, 11), seed=12)
    train, validation = split_engine_ids(range(1, 11), seed=12)
    assert np.array_equal(first_train, train)
    assert np.array_equal(first_validation, validation)
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(1, 11))


def test_windows_are_ordered_padded_and_never_cross_engine(toy_engines):
    batch = make_sequence_windows(
        toy_engines, window_size=3, sensor_columns=["sensor"], target_cap=125
    )
    assert batch.x.shape == (5, 3, 1)
    np.testing.assert_array_equal(batch.x[:3, :, 0], [[1, 1, 1], [1, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal(batch.x[3:, :, 0], [[100, 100, 100], [100, 100, 101]])
    np.testing.assert_array_equal(batch.unit, [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(batch.cycle, [1, 2, 3, 1, 2])
    np.testing.assert_array_equal(batch.y, [10, 9, 8, 125, 125])


def test_last_windows_align_with_final_cycle_labels(toy_engines):
    batch = make_sequence_windows(
        toy_engines,
        window_size=3,
        sensor_columns=["sensor"],
        target_cap=None,
        last_only=True,
    )
    np.testing.assert_array_equal(batch.unit, [1, 2])
    np.testing.assert_array_equal(batch.cycle, [3, 2])
    np.testing.assert_array_equal(batch.y, [8, 129])
    np.testing.assert_array_equal(batch.x[:, -1, 0], [3, 101])


def test_standardizer_uses_only_requested_training_engines(toy_engines):
    scaler = EngineStandardizer(["sensor"]).fit(toy_engines, training_units=[1])
    assert scaler.scaler_.mean_[0] == pytest.approx(2.0)
    transformed = scaler.transform(toy_engines)
    assert transformed.loc[toy_engines["unit"] == 2, "sensor"].mean() > 90
    np.testing.assert_array_equal(
        toy_engines["sensor"].to_numpy(), [1.0, 2.0, 3.0, 100.0, 101.0]
    )


def test_standardizer_converts_integer_sensor_columns(toy_engines):
    integer_frame = toy_engines.astype({"sensor": "int64"})
    transformed = EngineStandardizer(["sensor"]).fit(
        integer_frame, training_units=[1]
    ).transform(integer_frame)
    assert np.issubdtype(transformed["sensor"].dtype, np.floating)
    assert np.isfinite(transformed["sensor"]).all()


def test_unfitted_standardizer_and_invalid_windows_fail_explicitly(toy_engines):
    with pytest.raises(RuntimeError, match="must be fitted"):
        EngineStandardizer(["sensor"]).transform(toy_engines)
    with pytest.raises(ValueError, match="window_size"):
        make_sequence_windows(toy_engines, window_size=0, sensor_columns=["sensor"])
    with pytest.raises(ValueError, match="consecutive"):
        make_sequence_windows(
            toy_engines.assign(cycle=[1, 3, 4, 1, 2]),
            window_size=2,
            sensor_columns=["sensor"],
        )
