import importlib

import pytest
import torch

from .multi_margin_loss import multi_margin_loss

_RED = {0: "none", 1: "mean", 2: "sum"}


def _case(n=4, c=5):
    torch.manual_seed(0)
    return torch.randn(n, c), torch.randint(0, c, (n,))


@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("p", [1, 2])
def test_multi_margin_loss_matches_aten(p, reduction):
    x, t = _case()
    ref = torch.nn.functional.multi_margin_loss(x, t, p=p, reduction=_RED[reduction])
    tri = multi_margin_loss(x, t, p, 1.0, None, reduction)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_multi_margin_loss_weight_multiplies_row():
    # Probed on torch CPU: weight[target[i]] scales the whole row loss.
    x, t = _case()
    w = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])
    ref = torch.nn.functional.multi_margin_loss(x, t, p=1, weight=w, reduction="none")
    tri = multi_margin_loss(x, t, 1, 1.0, w, 0)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_multi_margin_loss_margin_kw():
    x, t = _case()
    ref = torch.nn.functional.multi_margin_loss(x, t, p=2, margin=0.5, reduction="none")
    tri = multi_margin_loss(x, t, 2, 0.5, None, 0)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_multi_margin_loss_empty_none():
    x = torch.empty(0, 5)
    t = torch.empty(0, dtype=torch.long)
    assert multi_margin_loss(x, t, 1, 1.0, None, 0).shape == (0,)


def test_multi_margin_loss_error_contracts():
    x, t = _case()
    with pytest.raises(ValueError):
        multi_margin_loss(x, t, 3, 1.0, None, 1)
    with pytest.raises(RuntimeError):
        multi_margin_loss(x, torch.tensor([1, 0, 5, 2]), 1, 1.0, None, 1)
    with pytest.raises(RuntimeError):
        multi_margin_loss(x, t.int(), 1, 1.0, None, 1)


def test_multi_margin_loss_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.multi_margin_loss")
    launches = []
    real_run = mod._multi_margin_loss_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._multi_margin_loss_kernel, "run", spy_run)
    x, t = _case()
    out = mod.multi_margin_loss(x, t, 1, 1.0, None, 1)
    assert len(launches) >= 1, "multi_margin_loss must launch the Triton kernel"
    ref = torch.nn.functional.multi_margin_loss(x, t, p=1, reduction="mean")
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)
