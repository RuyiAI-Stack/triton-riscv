import importlib
import inspect

import pytest
import torch

from .rrelu_ import rrelu_

_LOWER = 0.125
_UPPER = 0.33333333333333331


def test_rrelu_inplace_returns_self_eval_formula():
    torch.manual_seed(0)
    x = torch.randn(512) * 3.0
    slope = (_LOWER + _UPPER) * 0.5
    ref = torch.where(x >= 0, x, x * slope)

    out = rrelu_(x, _LOWER, _UPPER, training=False)

    assert out.data_ptr() == x.data_ptr(), "must return the same storage"
    torch.testing.assert_close(x, ref, rtol=0, atol=1e-7)


def test_rrelu_inplace_train_slopes():
    torch.manual_seed(0)
    x = -(torch.rand(5000) + 0.1)
    x_ref = x.clone()

    rrelu_(x, 0.25, 0.75, training=True)

    ratio = x / x_ref
    assert bool((ratio >= 0.25 - 1e-6).all()) and bool((ratio <= 0.75 + 1e-6).all())


def test_rrelu_inplace_generator_not_supported():
    x = torch.randn(8)
    with pytest.raises(AssertionError):
        rrelu_(x, training=True, generator=torch.Generator())


def test_rrelu_inplace_rejects_lower_above_upper_and_documents_generator():
    x = torch.randn(8)
    with pytest.raises(AssertionError):
        rrelu_(x, 0.5, 0.1, training=True)
    doc = inspect.getdoc(rrelu_) or ""
    assert "generator" in doc and "not supported" in doc


def test_rrelu_inplace_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.rrelu_")
    launches = []
    real_run = mod._rrelu_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._rrelu_kernel, "run", spy_run)
    x = torch.randn(1024)
    x_ref = x.clone()
    mod.rrelu_(x, 0.25, 0.75, training=True)
    assert len(launches) >= 1, "rrelu_ must launch the Triton kernel"
    pos = x_ref >= 0
    torch.testing.assert_close(x[pos], x_ref[pos], rtol=0, atol=0)
    ratio = x[~pos] / x_ref[~pos]
    assert bool((ratio >= 0.25 - 1e-6).all())
    assert bool((ratio <= 0.75 + 1e-6).all())


def test_rrelu_inplace_empty():
    x = torch.empty(0)
    out = rrelu_(x, training=True)
    assert out.data_ptr() == x.data_ptr() and out.numel() == 0
