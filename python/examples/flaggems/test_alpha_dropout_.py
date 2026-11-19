import importlib
import math

import torch

from .alpha_dropout_ import alpha_dropout_

_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_P = -_ALPHA * _SCALE


def _affine(p):
    keep = 1.0 - p
    a = 1.0 / math.sqrt(keep * (1.0 + p * _ALPHA_P * _ALPHA_P))
    b = -a * p * _ALPHA_P
    return a, b


def test_alpha_dropout_inplace_returns_self_and_eval_noop():
    torch.manual_seed(0)
    x = torch.randn(6, 8)
    original = x.clone()

    out = alpha_dropout_(x, 0.5, train=False)

    assert out.data_ptr() == x.data_ptr(), "must return the same storage"
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_alpha_dropout_inplace_p1_signed_zeros():
    torch.manual_seed(0)
    x = torch.randn(512)
    signs = torch.signbit(x)
    alpha_dropout_(x, 1.0, train=True)
    assert bool((x == 0).all())
    assert bool((torch.signbit(x) == signs).all())


def test_alpha_dropout_inplace_train_structure():
    torch.manual_seed(0)
    n = 20000
    p = 0.25
    x_ref = torch.randn(n)
    x = x_ref.clone()
    a, b = _affine(p)
    dropped_val = a * _ALPHA_P + b

    alpha_dropout_(x, p, train=True)

    is_dropped = (x - dropped_val).abs() < 1e-5
    is_kept = (x - (a * x_ref + b)).abs() < 1e-5
    assert bool((is_dropped | is_kept).all())


def test_alpha_dropout_inplace_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.alpha_dropout_")
    launches = []
    real_run = mod._alpha_dropout_inplace_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._alpha_dropout_inplace_kernel, "run", spy_run)
    x = torch.randn(1024)
    x_ref = x.clone()
    mod.alpha_dropout_(x, 0.5, train=True)
    assert len(launches) >= 1, "alpha_dropout_ must launch the Triton kernel"
    a, b = _affine(0.5)
    dropped_val = a * _ALPHA_P + b
    is_dropped = (x - dropped_val).abs() < 1e-5
    is_kept = (x - (a * x_ref + b)).abs() < 1e-5
    assert bool((is_dropped | is_kept).all()), (
        "spy path must keep the two-point contract"
    )


def test_alpha_dropout_inplace_empty():
    x = torch.empty(0)
    out = alpha_dropout_(x, 0.5, train=True)
    assert out.data_ptr() == x.data_ptr() and out.numel() == 0
    assert alpha_dropout_(x, 0.5, train=False).data_ptr() == x.data_ptr()
