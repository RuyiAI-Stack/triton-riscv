import importlib
import math

import pytest
import torch

from .dropout_ import dropout_


def test_dropout_eval_noop_returns_same_object():
    torch.manual_seed(0)
    x = torch.randn(6, 8)
    out = dropout_(x, 0.5, train=False)
    assert out is x, "eval must be a no-op returning the same object"


def test_dropout_p0_noop_and_p1_signed_zeros():
    torch.manual_seed(0)
    x = torch.randn(512)
    y = x.clone()
    dropout_(y, 0.0, train=True)
    torch.testing.assert_close(y, x, rtol=0, atol=0)

    signs = torch.signbit(x)
    dropout_(x, 1.0, train=True)
    assert bool((x == 0).all())
    assert bool((torch.signbit(x) == signs).all())


def test_dropout_train_structure_scaling_and_rate():
    torch.manual_seed(0)
    n = 50000
    p = 0.4
    x = torch.randn(n)

    out = dropout_(x.clone(), p, train=True)

    nonzero = out != 0
    # Survivors are scaled by exactly 1/(1-p) within fp32 rounding.
    ratio_ok = (out[nonzero] / x[nonzero] - 1.0 / (1 - p)).abs() < 1e-5
    assert bool(ratio_ok.all()), "kept elements must be x / (1 - p)"
    kept_frac = float(nonzero.float().mean())
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(kept_frac - (1 - p)) < 4 * sigma + 1e-3, f"keep rate {kept_frac}"
    # Unbiasedness: the mean survives the rescaling.
    assert abs(float(out.mean()) - float(x.mean())) < 4 * sigma * float(x.abs().mean())


def test_dropout_deterministic_under_same_seed():
    torch.manual_seed(0)
    x = torch.randn(4096)
    torch.manual_seed(1234)
    first = dropout_(x.clone(), 0.5, train=True)
    torch.manual_seed(1234)
    second = dropout_(x.clone(), 0.5, train=True)
    assert torch.equal(first, second)


def test_dropout_rejects_noncontiguous_input():
    # Mirrors the upstream requirement: the in-place path asserts a
    # contiguous buffer (ATen's in-place dropout tolerates strided inputs,
    # but the port keeps the upstream contract).
    x = torch.randn(4, 4).t()
    with pytest.raises(AssertionError):
        dropout_(x, 0.5, train=True)


def test_dropout_empty():
    empty = torch.empty(0)
    out = dropout_(empty, 0.5, train=True)
    assert out is empty and out.shape == (0,)


def test_dropout_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.dropout_")
    launches = []
    real_run = mod._dropout_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._dropout_kernel, "run", spy_run)
    x = torch.randn(1024)
    x_ref = x.clone()
    mod.dropout_(x, 0.5, train=True)
    assert len(launches) >= 1, "dropout_ must launch the Triton kernel"
    nonzero = x != 0
    assert bool(((x[nonzero] / x_ref[nonzero] - 2.0).abs() < 1e-5).all())
    assert bool((x[~nonzero] == 0).all())
