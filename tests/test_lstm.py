"""Shape, gradient, and configuration tests for the LSTM regressor."""

import pytest
import torch

from src.lstm import LSTMRegressor


def test_lstm_forward_and_gradients():
    model = LSTMRegressor(input_size=4, hidden_size=12, num_layers=2, dropout=0.1)
    sequences = torch.randn(5, 30, 4)
    predictions = model(sequences)
    assert predictions.shape == (5,)
    assert torch.isfinite(predictions).all()

    predictions.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_lstm_accepts_single_layer_and_serializable_config():
    model = LSTMRegressor(input_size=3, hidden_size=8, num_layers=1, dropout=0.25)
    assert model.get_config() == {
        "input_size": 3,
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.25,
    }


@pytest.mark.parametrize(
    "config",
    [
        {"input_size": 0},
        {"hidden_size": 0},
        {"num_layers": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
    ],
)
def test_lstm_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        LSTMRegressor(**config)


def test_lstm_rejects_invalid_tensor_shapes():
    model = LSTMRegressor(input_size=3, hidden_size=8, num_layers=1)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(5, 3))
    with pytest.raises(ValueError, match="expected 3 input features"):
        model(torch.randn(5, 10, 4))
