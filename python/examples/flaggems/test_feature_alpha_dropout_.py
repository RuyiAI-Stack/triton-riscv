import importlib
import math

import torch

from .feature_alpha_dropout_ import feature_alpha_dropout_

_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_P = -_ALPHA * _SCALE


def _affine(p):
    keep = 1.0 - p
    a = 1.0 / math.sqrt(keep * (1.0 + p * _ALPHA_P * _ALPHA_P))
    b = -a * p * _ALPHA_P
    return a, b


def test_feature_alpha_dropout_inplace_returns_self_eval_noop():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4, 4)
    original = x.clone()

    out = feature_alpha_dropout_(x, 0.5, train=False)

    assert out.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_feature_alpha_dropout_inplace_channel_structure():
    p = 0.5
    torch.manual_seed(0)
    x = torch.randn(4, 256, 2, 2)
    x_ref = x.clone()
    a, b = _affine(p)
    dropped_val = a * _ALPHA_P + b

    feature_alpha_dropout_(x, p, train=True)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            chan = x[i, j]
            chan_dropped = bool((chan - dropped_val).abs().max() < 1e-5)
            chan_kept = bool((chan - (a * x_ref[i, j] + b)).abs().max() < 1e-5)
            assert chan_dropped or chan_kept


def test_feature_alpha_dropout_inplace_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.feature_alpha_dropout_")
    launches = []
    real_run = mod._feature_alpha_dropout_inplace_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._feature_alpha_dropout_inplace_kernel, "run", spy_run)
    x = torch.randn(2, 8, 4, 4)
    x_ref = x.clone()
    mod.feature_alpha_dropout_(x, 0.5, train=True)
    assert len(launches) >= 1, "feature_alpha_dropout_ must launch the Triton kernel"
    a, b = _affine(0.5)
    dropped_val = a * _ALPHA_P + b
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            chan = x[i, j]
            chan_dropped = bool((chan - dropped_val).abs().max() < 1e-5)
            chan_kept = bool((chan - (a * x_ref[i, j] + b)).abs().max() < 1e-5)
            assert chan_dropped or chan_kept


def test_feature_alpha_dropout_inplace_empty():
    x = torch.empty(0, 8, 2, 2)
    out = feature_alpha_dropout_(x, 0.5, train=True)
    assert out.data_ptr() == x.data_ptr() and out.shape == (0, 8, 2, 2)
