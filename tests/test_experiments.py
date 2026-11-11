"""Tests for Phase 2 validation configuration and selection rules."""

import pytest

from src.config import RUL_CAP
from src.experiments import (
    phase2_experiment_specs,
    select_best_experiment,
    validation_cutoff_cycles,
)


def test_experiment_sweep_covers_window_lengths_and_regularization():
    specs = phase2_experiment_specs()
    assert {spec["window_size"] for spec in specs} == {20, 30, 50}
    window_30 = [spec for spec in specs if spec["window_size"] == 30]
    assert len({spec["dropout"] for spec in window_30}) > 1
    assert len({spec["weight_decay"] for spec in window_30}) > 1


def test_model_selection_uses_validation_rmse_only():
    results = [
        {"name": "a", "validation_rmse_capped": 12.0, "test_rmse": 1.0},
        {"name": "b", "validation_rmse_capped": 9.0, "test_rmse": 50.0},
    ]
    assert select_best_experiment(results)["name"] == "b"
    with pytest.raises(ValueError, match="at least one"):
        select_best_experiment([])


def test_validation_cutoffs_are_deterministic_and_pre_failure():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"unit": unit, "cycle": cycle, "RUL": final_cycle - cycle}
            for unit, final_cycle in [(1, 140), (2, 170)]
            for cycle in range(1, final_cycle + 1)
        ]
    )
    cutoffs = validation_cutoff_cycles(frame, seed=7)
    assert cutoffs == validation_cutoff_cycles(frame, seed=7)
    for unit, cycle in cutoffs.items():
        final_cycle = int(frame.loc[frame["unit"] == unit, "cycle"].max())
        remaining_life = final_cycle - cycle
        assert 1 <= remaining_life <= RUL_CAP
        assert cycle < final_cycle
