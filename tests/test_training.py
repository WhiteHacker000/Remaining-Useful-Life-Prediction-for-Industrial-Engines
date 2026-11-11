"""Integration tests for training, checkpoint restoration, and prediction."""

import numpy as np
import pytest
import torch

from src.sequence_data import SequenceBatch
from src.training import TrainingConfig, predict_lstm, train_lstm


def make_batch(units, samples_per_engine=4):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(len(units) * samples_per_engine, 6, 2)).astype(np.float32)
    y = rng.uniform(0, 20, size=len(x)).astype(np.float32)
    unit = np.repeat(units, samples_per_engine)
    cycle = np.tile(np.arange(1, samples_per_engine + 1), len(units))
    return SequenceBatch(x=x, y=y, unit=unit, cycle=cycle)


def test_training_saves_history_checkpoint_and_restores_best_weights(tmp_path):
    train = make_batch([1, 2])
    validation = make_batch([3])
    config = TrainingConfig(
        epochs=4, batch_size=4, patience=3, seed=13, device="cpu"
    )
    result = train_lstm(
        train,
        validation,
        tmp_path / "model.pt",
        model_config={"input_size": 2, "hidden_size": 8, "num_layers": 1},
        config=config,
    )

    assert result.checkpoint_path.exists()
    assert result.history_path.exists()
    assert 1 <= result.best_epoch <= len(result.history) <= config.epochs
    assert result.history[result.best_epoch - 1]["validation_loss"] == pytest.approx(
        result.best_validation_loss
    )
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    restored = type(result.model)(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    np.testing.assert_allclose(
        predict_lstm(result.model, validation),
        predict_lstm(restored, validation),
        rtol=0,
        atol=1e-6,
    )
    assert len(predict_lstm(result.model, validation)) == len(validation.y)


def test_training_rejects_engine_overlap(tmp_path):
    same_engines = make_batch([1, 2])
    with pytest.raises(ValueError, match="disjoint engines"):
        train_lstm(
            same_engines,
            make_batch([2]),
            tmp_path / "should-not-exist.pt",
            model_config={"input_size": 2, "hidden_size": 4, "num_layers": 1},
            config=TrainingConfig(epochs=1, device="cpu"),
        )
    assert not (tmp_path / "should-not-exist.pt").exists()


def test_final_fit_uses_all_training_engines_without_validation(tmp_path):
    training = make_batch([1, 2, 3])
    result = train_lstm(
        training,
        None,
        tmp_path / "final.pt",
        model_config={"input_size": 2, "hidden_size": 4, "num_layers": 1},
        config=TrainingConfig(epochs=2, batch_size=4, device="cpu"),
    )
    assert result.best_epoch == 2
    assert result.best_validation_loss is None
    assert all(epoch["validation_loss"] is None for epoch in result.history)


def test_training_configuration_rejects_invalid_values():
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="optimizer"):
        TrainingConfig(optimizer="sgd")
