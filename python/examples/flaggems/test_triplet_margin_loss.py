import importlib

import pytest
import torch

from .triplet_margin_loss import triplet_margin_loss

_RED = {0: "none", 1: "mean", 2: "sum"}


def _case(d=8):
    torch.manual_seed(0)
    return (torch.randn(4, d) for _ in range(3))


@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("swap", [False, True])
def test_triplet_margin_loss_matches_aten(p, swap, reduction):
    a, pos, neg = _case()
    ref = torch.nn.functional.triplet_margin_loss(
        a, pos, neg, margin=1.0, p=p, eps=1e-6, swap=swap, reduction=_RED[reduction]
    )
    tri = triplet_margin_loss(a, pos, neg, 1.0, p, 1e-6, swap, reduction)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_triplet_margin_loss_custom_margin_and_p_norm():
    a, pos, neg = _case()
    ref = torch.nn.functional.triplet_margin_loss(
        a, pos, neg, margin=0.7, p=2, eps=1e-3, swap=True, reduction="none"
    )
    tri = triplet_margin_loss(a, pos, neg, 0.7, 2, 1e-3, True, 0)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_triplet_margin_loss_empty_none():
    e = torch.empty(0, 8)
    out = triplet_margin_loss(e, e, e, 1.0, 2, 1e-6, False, 0)
    assert out.shape == (0,)


def test_triplet_margin_loss_margin_must_be_positive():
    a, pos, neg = _case()
    with pytest.raises(ValueError):
        triplet_margin_loss(a, pos, neg, 0.0, 2, 1e-6, False, 1)
    with pytest.raises(ValueError):
        triplet_margin_loss(a, pos, neg, -1.0, 2, 1e-6, False, 1)


def test_triplet_margin_loss_rejects_unsupported_p():
    # ATen accepts any real p, but pow does not lower on this toolchain, so
    # this port supports p in {1, 2} only; the rejection is part of the
    # contract (documented in the docstring).
    a, pos, neg = _case()
    with pytest.raises(ValueError):
        triplet_margin_loss(a, pos, neg, 1.0, 3, 1e-6, False, 1)


def test_triplet_margin_loss_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.triplet_margin_loss")
    launches = []
    real_run = mod._triplet_margin_loss_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._triplet_margin_loss_kernel, "run", spy_run)
    a, pos, neg = _case()
    out = mod.triplet_margin_loss(a, pos, neg, 1.0, 2, 1e-6, True, 1)
    assert len(launches) >= 1, "triplet_margin_loss must launch the Triton kernel"
    ref = torch.nn.functional.triplet_margin_loss(a, pos, neg, swap=True)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-5)
