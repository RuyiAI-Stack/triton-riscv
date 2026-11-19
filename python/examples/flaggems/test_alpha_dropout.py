import importlib
import math

import torch

from .alpha_dropout import alpha_dropout

_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_P = -_ALPHA * _SCALE


def _affine(p):
    # SELU affine parameters preserving mean 0 / variance 1 (upstream form).
    keep = 1.0 - p
    a = 1.0 / math.sqrt(keep * (1.0 + p * _ALPHA_P * _ALPHA_P))
    b = -a * p * _ALPHA_P
    return a, b


def test_alpha_dropout_eval_is_identity():
    torch.manual_seed(0)
    x = torch.randn(6, 8) * 3.0
    out = alpha_dropout(x, 0.5, train=False)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    # Probed on torch CPU: eval is a pass-through returning the input object.
    assert out.data_ptr() == x.data_ptr()


def test_alpha_dropout_p0_identity_and_p1_signed_zeros():
    torch.manual_seed(0)
    x = torch.randn(512)
    torch.testing.assert_close(alpha_dropout(x, 0.0, train=True), x, rtol=0, atol=0)

    out = alpha_dropout(x, 1.0, train=True)
    assert bool((out == 0).all()), "p=1 must zero everything"
    # Probed on torch CPU: the signed zero is preserved (signbit follows x).
    assert bool((torch.signbit(out) == torch.signbit(x)).all())


def test_alpha_dropout_train_two_point_structure_and_rate():
    torch.manual_seed(0)
    n = 50000
    p = 0.4
    x = torch.randn(n)
    a, b = _affine(p)
    dropped_val = a * _ALPHA_P + b

    out = alpha_dropout(x, p, train=True)

    is_dropped = (out - dropped_val).abs() < 1e-5
    is_kept = (out - (a * x + b)).abs() < 1e-5
    assert bool((is_dropped | is_kept).all()), (
        "every element must be either the dropped constant or a*x + b"
    )
    dropped_frac = float(is_dropped.float().mean())
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(dropped_frac - p) < 4 * sigma + 1e-3, f"drop rate {dropped_frac}"


def test_alpha_dropout_deterministic_under_same_seed():
    torch.manual_seed(0)
    x = torch.randn(4096)
    torch.manual_seed(1234)
    first = alpha_dropout(x, 0.5, train=True)
    torch.manual_seed(1234)
    second = alpha_dropout(x, 0.5, train=True)
    assert torch.equal(first, second)


def test_alpha_dropout_empty():
    empty = torch.empty(0)
    assert alpha_dropout(empty, 0.5, train=True).shape == (0,)


def test_alpha_dropout_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.alpha_dropout")
    launches = []
    real_run = mod._alpha_dropout_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._alpha_dropout_kernel, "run", spy_run)
    x = torch.randn(1024)
    out = mod.alpha_dropout(x, 0.5, train=True)
    assert len(launches) >= 1, "alpha_dropout must launch the Triton kernel"
    a, b = _affine(0.5)
    dropped_val = a * _ALPHA_P + b
    is_dropped = (out - dropped_val).abs() < 1e-5
    is_kept = (out - (a * x + b)).abs() < 1e-5
    assert bool((is_dropped | is_kept).all()), (
        "spy path must keep the two-point contract"
    )
