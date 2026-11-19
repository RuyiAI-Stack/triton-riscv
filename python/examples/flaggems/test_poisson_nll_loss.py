import importlib
import math

import pytest
import torch

from .poisson_nll_loss import poisson_nll_loss

_RED = {0: "none", 1: "mean", 2: "sum"}


def _case(log_input):
    torch.manual_seed(0)
    if log_input:
        x = torch.randn(3, 5)
        t = torch.rand(3, 5) * 3.0
    else:
        x = torch.rand(3, 5) * 2.0 + 0.1
        t = torch.rand(3, 5) * 3.0
    return x, t


@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("log_input", [True, False])
def test_poisson_nll_loss_matches_aten(log_input, full, reduction):
    x, t = _case(log_input)
    ref = torch.nn.functional.poisson_nll_loss(
        x, t, log_input=log_input, full=full, eps=1e-8, reduction=_RED[reduction]
    )
    tri = poisson_nll_loss(x, t, log_input, full, 1e-8, reduction)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-5)


def test_poisson_nll_loss_stirling_gate():
    # Probed on torch CPU: full=True adds t*ln(t) - t + 0.5*ln(2*pi*t) only
    # where t > 1 (t = 2.0 adds it, t = 0.5 and t = 1.0 do not).
    x = torch.tensor([[2.0, 2.0, 2.0]])
    t = torch.tensor([[2.0, 0.5, 1.0]])
    base = poisson_nll_loss(x, t, True, False, 1e-8, 0)
    full = poisson_nll_loss(x, t, True, True, 1e-8, 0)
    stirling = 2.0 * math.log(2.0) - 2.0 + 0.5 * math.log(2 * math.pi * 2.0)
    torch.testing.assert_close(
        full - base, torch.tensor([[stirling, 0.0, 0.0]]), rtol=0.0, atol=1e-5
    )


def test_poisson_nll_loss_empty_none():
    x = torch.empty(0)
    t = torch.empty(0)
    assert poisson_nll_loss(x, t, True, False, 1e-8, 0).shape == (0,)


def test_poisson_nll_loss_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.poisson_nll_loss")
    launches = []
    real_run = mod._poisson_nll_loss_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._poisson_nll_loss_kernel, "run", spy_run)
    x, t = _case(True)
    out = mod.poisson_nll_loss(x, t, True, True, 1e-8, 1)
    assert len(launches) >= 1, "poisson_nll_loss must launch the Triton kernel"
    ref = torch.nn.functional.poisson_nll_loss(
        x, t, log_input=True, full=True, reduction="mean"
    )
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-5)
