import importlib
import inspect

import pytest
import torch

from .rrelu import rrelu

_LOWER = 0.125
_UPPER = 0.33333333333333331


def test_rrelu_eval_matches_formula():
    torch.manual_seed(0)
    x = torch.randn(512) * 3.0
    slope = (_LOWER + _UPPER) * 0.5
    ref = torch.where(x >= 0, x, x * slope)

    out = rrelu(x, _LOWER, _UPPER, training=False)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-7)


def test_rrelu_train_slopes_in_range_and_positives_untouched():
    torch.manual_seed(0)
    neg = -(torch.rand(10000) + 0.1)
    pos = torch.rand(64) + 0.1
    x = torch.cat([neg, pos])

    out = rrelu(x, _LOWER, _UPPER, training=True)

    ratio = out[: neg.numel()] / neg
    assert bool((ratio >= _LOWER - 1e-6).all()) and bool((ratio <= _UPPER + 1e-6).all())
    # The slope must actually vary (it is sampled), not collapse to a constant.
    assert float(ratio.max() - ratio.min()) > 0.01
    torch.testing.assert_close(out[neg.numel() :], pos, rtol=0, atol=0)


def test_rrelu_strided_input_matches_formula():
    # Non-contiguous inputs are the trap that *_like(self) helpers hide:
    # the slope buffer must stay contiguous while x is flattened.
    torch.manual_seed(0)
    x = torch.randn(8, 32).t()
    assert not x.is_contiguous()
    slope = (_LOWER + _UPPER) * 0.5
    ref = torch.where(x >= 0, x, x * slope)
    out = rrelu(x, _LOWER, _UPPER, training=False)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-7)

    neg = -(torch.rand(4, 64).t() + 0.1)
    out_t = rrelu(neg, _LOWER, _UPPER, training=True)
    ratio = out_t / neg
    assert bool((ratio >= _LOWER - 1e-6).all()) and bool((ratio <= _UPPER + 1e-6).all())


def test_rrelu_deterministic_under_same_seed():
    torch.manual_seed(0)
    x = -torch.rand(4096) - 0.1
    torch.manual_seed(1234)
    first = rrelu(x, training=True)
    torch.manual_seed(1234)
    second = rrelu(x, training=True)
    assert torch.equal(first, second)


def test_rrelu_generator_not_supported():
    # Mirrors upstream: the generator argument is asserted to be None.
    x = torch.randn(8)
    with pytest.raises(AssertionError):
        rrelu(x, training=True, generator=torch.Generator())
    doc = inspect.getdoc(rrelu) or ""
    assert "generator" in doc and "not supported" in doc


def test_rrelu_rejects_lower_above_upper():
    # Mirrors the upstream argument check (assert lower <= upper); ATen
    # raises RuntimeError here, the port keeps the upstream contract.
    x = torch.randn(8)
    with pytest.raises(AssertionError):
        rrelu(x, 0.5, 0.1, training=True)


def test_rrelu_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.rrelu")
    launches = []
    real_run = mod._rrelu_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._rrelu_kernel, "run", spy_run)
    x = torch.randn(1024)
    out = mod.rrelu(x, 0.25, 0.75, training=True)
    assert len(launches) >= 1, "rrelu must launch the Triton kernel"
    pos = x >= 0
    torch.testing.assert_close(out[pos], x[pos], rtol=0, atol=0)
    ratio = out[~pos] / x[~pos]
    assert bool((ratio >= 0.25 - 1e-6).all())
    assert bool((ratio <= 0.75 + 1e-6).all())


def test_rrelu_empty():
    empty = torch.empty(0)
    assert rrelu(empty, training=False).shape == (0,)
    assert rrelu(empty, training=True).shape == (0,)
