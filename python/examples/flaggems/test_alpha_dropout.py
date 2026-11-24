import pytest
import torch
import torch.nn.functional as F

from .alpha_dropout import alpha_dropout


def test_alpha_dropout_eval_returns_clone():
    x = torch.randn(1024, dtype=torch.float32, device="cpu")

    out = alpha_dropout(x, p=0.5, train=False)
    ref = F.alpha_dropout(x, p=0.5, training=False)

    assert out is not x
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_alpha_dropout_deterministic_probabilities_match_torch(p):
    x = torch.randn(1023, dtype=torch.float32, device="cpu")

    out = alpha_dropout(x, p=p, train=True)
    ref = F.alpha_dropout(x, p=p, training=True)

    torch.testing.assert_close(out, ref)


def test_alpha_dropout_train_properties():
    torch.manual_seed(0)
    x = torch.randn(32768, dtype=torch.float32, device="cpu")

    out = alpha_dropout(x, p=0.2, train=True)
    torch.manual_seed(0)
    ref = F.alpha_dropout(x, p=0.2, training=True)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.mean(), ref.mean(), rtol=0, atol=5e-2)
    torch.testing.assert_close(out.var(), ref.var(), rtol=0, atol=5e-2)
