"""Tests for the Phase 2 sequence models. Run with:  pytest -q"""

import pytest

torch = pytest.importorskip("torch")
from src.dl_models import TCNEncoder, make_dl_model  # noqa: E402


@pytest.mark.parametrize("name", ["lstm", "tcn", "transformer"])
def test_models_produce_one_output_per_window(name):
    x = torch.randn(8, 30, 15)
    x[:, :, -1] = 1.0
    assert make_dl_model(name, n_in=15, window=30)(x).shape == (8,)


def test_tcn_is_causal_and_covers_the_window():
    enc = TCNEncoder(n_in=15, window=30)
    assert enc.receptive_field >= 30
    enc.eval()
    x = torch.randn(1, 30, 15)
    changed = x.clone()
    changed[:, -1] += 5.0                       # change only the last (current) cycle
    with torch.no_grad():
        a = enc.net(x.transpose(1, 2))[:, :, :-1]
        b = enc.net(changed.transpose(1, 2))[:, :, :-1]
    assert torch.allclose(a, b)                 # earlier outputs cannot see the future


def test_transformer_ignores_padded_time_steps():
    """Padded steps are removed from attention, so their values must not change the output."""
    model = make_dl_model("transformer", n_in=15, window=30).eval()
    x = torch.randn(2, 30, 15)
    x[:, :, -1] = 1.0
    x[:, :10, -1] = 0.0                         # first 10 steps are padding
    changed = x.clone()
    changed[:, :10, :-1] += 5.0                 # change only the padded sensor values
    with torch.no_grad():
        assert torch.allclose(model(x), model(changed), atol=1e-5)


@pytest.mark.parametrize("name", ["lstm", "tcn", "transformer"])
def test_models_depend_on_the_order_of_cycles(name):
    """A sequence model must read the window as a sequence: reversing time changes its output."""
    torch.manual_seed(0)
    model = make_dl_model(name, n_in=15, window=30).eval()
    x = torch.randn(4, 30, 15)
    x[:, :, -1] = 1.0
    with torch.no_grad():
        assert (model(x) - model(x.flip(1))).abs().max() > 1e-4
