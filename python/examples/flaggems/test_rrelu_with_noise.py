import importlib
import inspect

import pytest
import torch

from .rrelu_with_noise import rrelu_with_noise

_LOWER = 0.125
_UPPER = 0.33333333333333331


def _mixed_input():
    torch.manual_seed(0)
    neg = -(torch.rand(256) + 0.1)
    pos = torch.rand(256) + 0.1
    x = torch.cat([neg, pos])
    return x, neg, pos


def test_rrelu_with_noise_train_writes_noise_and_multiplies():
    x, neg, pos = _mixed_input()
    noise = torch.ones_like(x) * 0.5

    out = rrelu_with_noise(x, noise, _LOWER, _UPPER, training=True)

    # x >= 0: noise slot reset to exactly 1, output unchanged.
    assert bool((noise[x >= 0] == 1.0).all())
    torch.testing.assert_close(out[x >= 0], x[x >= 0], rtol=0, atol=0)
    # x < 0: the noise slot holds a sample in [lower, upper] and out = x * noise.
    sampled = noise[x < 0]
    assert bool((sampled >= _LOWER - 1e-6).all()) and bool(
        (sampled <= _UPPER + 1e-6).all()
    )
    torch.testing.assert_close(out, x * noise, rtol=0, atol=0)


def test_rrelu_with_noise_eval_formula_and_noise_untouched():
    x, _, _ = _mixed_input()
    noise = torch.rand_like(x) + 0.25
    noise_before = noise.clone()

    out = rrelu_with_noise(x, noise, _LOWER, _UPPER, training=False)

    slope = (_LOWER + _UPPER) * 0.5
    ref = torch.where(x >= 0, x, x * slope)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-7)
    assert torch.equal(noise, noise_before), "eval must not touch the noise tensor"


def test_rrelu_with_noise_rejects_bad_noise_shape():
    x = torch.randn(8)
    with pytest.raises(AssertionError):
        rrelu_with_noise(x, torch.ones(7), _LOWER, _UPPER, True)


def test_rrelu_with_noise_deterministic_under_same_seed():
    torch.manual_seed(0)
    x = -torch.rand(2048) - 0.1
    torch.manual_seed(1234)
    first = rrelu_with_noise(x, torch.ones_like(x), _LOWER, _UPPER, True)
    torch.manual_seed(1234)
    second = rrelu_with_noise(x, torch.ones_like(x), _LOWER, _UPPER, True)
    assert torch.equal(first, second)


def test_rrelu_with_noise_generator_not_supported():
    x = torch.randn(8)
    noise = torch.ones_like(x)
    with pytest.raises(AssertionError):
        rrelu_with_noise(x, noise, _LOWER, _UPPER, True, generator=torch.Generator())
    doc = inspect.getdoc(rrelu_with_noise) or ""
    assert "generator" in doc and "not supported" in doc


def test_rrelu_with_noise_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.rrelu_with_noise")
    launches = []
    real_run = mod._rrelu_with_noise_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._rrelu_with_noise_kernel, "run", spy_run)
    x = torch.randn(1024)
    noise = torch.ones_like(x)
    out = mod.rrelu_with_noise(x, noise, _LOWER, _UPPER, True)
    assert len(launches) >= 1, "rrelu_with_noise must launch the Triton kernel"
    torch.testing.assert_close(out, x * noise, rtol=0, atol=0)
    assert bool((noise[x >= 0] == 1.0).all())


def test_rrelu_with_noise_empty():
    x = torch.empty(0)
    noise = torch.empty(0)
    out = rrelu_with_noise(x, noise, _LOWER, _UPPER, True)
    assert out.shape == (0,) and noise.shape == (0,)
