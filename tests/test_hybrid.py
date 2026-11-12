"""Tests for the Phase 2 hybrid model. Run with:  pytest -q"""

import pytest

torch = pytest.importorskip("torch")
from src.dl_models import make_dl_model  # noqa: E402


@pytest.mark.parametrize("name", ["features", "hybrid-lstm", "hybrid-tcn", "hybrid-transformer"])
def test_hybrid_and_feature_models_produce_one_output_per_window(name):
    x = torch.randn(8, 30, 15)
    x[:, :, -1] = 1.0
    f = torch.randn(8, 88)
    assert make_dl_model(name, n_in=15, window=30, n_feat=88)(x, f).shape == (8,)


def test_hybrid_uses_both_branches():
    model = make_dl_model("hybrid-lstm", n_in=15, window=30, n_feat=88).eval()
    x, f = torch.randn(4, 30, 15), torch.randn(4, 88)
    x[:, :, -1] = 1.0
    with torch.no_grad():
        base = model(x, f)
        assert not torch.allclose(base, model(x, f + 1.0))          # depends on the features
        assert not torch.allclose(base, model(x + 1.0, f))          # depends on the sequence
