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
