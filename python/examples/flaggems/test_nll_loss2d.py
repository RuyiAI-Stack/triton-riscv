import importlib

import pytest
import torch

from .nll_loss2d import nll_loss2d

_RED = {0: "none", 1: "mean", 2: "sum"}


def _case():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4)
    t = torch.randint(0, 3, (2, 4, 4))
    t[0, 0, 0] = -100
    t[1, 2, 3] = -100
    return x, t


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_nll_loss2d_matches_aten(reduction):
    x, t = _case()
    ref = torch.nn.functional.nll_loss(x, t, reduction=_RED[reduction])
    tri = nll_loss2d(x, t, None, reduction, -100)
    assert tri.dtype == ref.dtype and tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6, equal_nan=True)


def test_nll_loss2d_weighted_mean():
    x, t = _case()
    w = torch.tensor([1.0, 2.0, 4.0])
    ref = torch.nn.functional.nll_loss(x, t, weight=w, reduction="mean")
    tri = nll_loss2d(x, t, w, 1, -100)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_nll_loss2d_ignored_positions_are_zero_in_none():
    # Probed on torch CPU: reduction=none writes 0 at ignored positions.
    x, t = _case()
    out = nll_loss2d(x, t, None, 0, -100)
    assert out[0, 0, 0].item() == 0.0 and out[1, 2, 3].item() == 0.0


def test_nll_loss2d_all_ignored_mean_is_nan():
    x, t = _case()
    out = nll_loss2d(x, torch.full_like(t, -100), None, 1, -100)
    assert torch.isnan(out).item() is True


def test_nll_loss2d_empty_spatial_none():
    x = torch.randn(2, 3, 0, 0)
    t = torch.empty(2, 0, 0, dtype=torch.long)
    out = nll_loss2d(x, t, None, 0, -100)
    assert out.shape == (2, 0, 0)


def test_nll_loss2d_error_contracts():
    x, t = _case()
    with pytest.raises(IndexError):
        nll_loss2d(x, torch.full_like(t, 7), None, 1, -100)
    with pytest.raises(RuntimeError):
        nll_loss2d(x, t.int(), None, 1, -100)


def test_nll_loss2d_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.nll_loss2d")
    launches = []
    real_run = mod._nll_loss2d_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._nll_loss2d_kernel, "run", spy_run)
    x, t = _case()
    out = mod.nll_loss2d(x, t, None, 1, -100)
    assert len(launches) >= 1, "nll_loss2d must launch the Triton kernel"
    ref = torch.nn.functional.nll_loss(x, t, reduction="mean")
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6, equal_nan=True)


def test_nll_loss2d_custom_ignore_index():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 3)
    t = torch.randint(0, 3, (2, 3, 3))
    t[0, 1, 1] = 7
    t[1, 2, 0] = 7
    for reduction in (0, 1, 2):
        ref = torch.nn.functional.nll_loss(
            x, t, reduction=_RED[reduction], ignore_index=7
        )
        tri = nll_loss2d(x, t, None, reduction, 7)
        torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6, equal_nan=True)


def test_nll_loss2d_empty_batch():
    # Probed on torch CPU: an empty batch keeps reduction=none at (0, H, W),
    # mean is nan (0/0) and sum is exactly 0.
    x = torch.randn(0, 3, 4, 4)
    t = torch.empty(0, 4, 4, dtype=torch.long)
    for reduction in (0, 1, 2):
        ref = torch.nn.functional.nll_loss(x, t, reduction=_RED[reduction])
        tri = nll_loss2d(x, t, None, reduction, -100)
        assert tri.shape == ref.shape
        torch.testing.assert_close(tri, ref, rtol=0.0, atol=0.0, equal_nan=True)
