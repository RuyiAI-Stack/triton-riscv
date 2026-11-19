import importlib
import inspect

import pytest
import torch

from .rrelu_with_noise_functional import rrelu_with_noise_functional

_LOWER = 0.125
_UPPER = 0.33333333333333331


def _mixed_input():
    torch.manual_seed(0)
    neg = -(torch.rand(128) + 0.1)
    pos = torch.rand(128) + 0.1
    return torch.cat([neg, pos])


def test_functional_train_returns_mutated_noise_tuple():
    x = _mixed_input()
    noise = torch.ones_like(x) * 0.5

    output, ret_noise = rrelu_with_noise_functional(
        x, noise, _LOWER, _UPPER, training=True
    )

    # The contract is that the caller's noise tensor is written through
    # (ATen mutates it in place); the returned tensor carries the same
    # values but is not required to be the same Python object on the train
    # path (the eval path returns the argument itself).
    assert torch.equal(ret_noise, noise)
    assert bool((noise[x >= 0] == 1.0).all()), "the caller's noise is mutated"
    assert bool((ret_noise[x >= 0] == 1.0).all())
    sampled = ret_noise[x < 0]
    assert bool((sampled >= _LOWER - 1e-6).all()) and bool(
        (sampled <= _UPPER + 1e-6).all()
    )
    torch.testing.assert_close(output, x * ret_noise, rtol=0, atol=0)


def test_functional_eval_returns_unmodified_noise():
    x = _mixed_input()
    noise = torch.rand_like(x) + 0.25
    noise_before = noise.clone()

    output, ret_noise = rrelu_with_noise_functional(
        x, noise, _LOWER, _UPPER, training=False
    )

    assert ret_noise is noise
    assert torch.equal(noise, noise_before), "eval returns the noise untouched"
    slope = (_LOWER + _UPPER) * 0.5
    ref = torch.where(x >= 0, x, x * slope)
    torch.testing.assert_close(output, ref, rtol=0, atol=1e-7)


def test_functional_rejects_bad_noise_shape_and_generator():
    x = torch.randn(8)
    with pytest.raises(AssertionError):
        rrelu_with_noise_functional(x, torch.ones(7), training=True)
    with pytest.raises(AssertionError):
        rrelu_with_noise_functional(
            x, torch.ones_like(x), training=True, generator=torch.Generator()
        )
    doc = inspect.getdoc(rrelu_with_noise_functional) or ""
    assert "generator" in doc and "not supported" in doc


def test_functional_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.rrelu_with_noise_functional")
    launches = []
    real_run = mod._rrelu_with_noise_functional_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._rrelu_with_noise_functional_kernel, "run", spy_run)
    x = torch.randn(1024)
    noise = torch.ones_like(x)
    out, ret_noise = mod.rrelu_with_noise_functional(
        x, noise, _LOWER, _UPPER, training=True
    )
    assert len(launches) >= 1, "the functional path must launch the Triton kernel"
    torch.testing.assert_close(out, x * ret_noise, rtol=0, atol=0)


def test_functional_empty():
    x = torch.empty(0)
    noise = torch.empty(0)
    out, ret_noise = rrelu_with_noise_functional(x, noise, _LOWER, _UPPER, True)
    assert out.shape == (0,) and ret_noise.shape == (0,)
