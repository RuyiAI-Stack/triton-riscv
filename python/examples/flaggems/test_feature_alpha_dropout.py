import importlib
import math

import torch

from .feature_alpha_dropout import feature_alpha_dropout

_ALPHA = 1.6732632423543772848170429916717
_SCALE = 1.0507009873554804934193349852946
_ALPHA_P = -_ALPHA * _SCALE


def _affine(p):
    keep = 1.0 - p
    a = 1.0 / math.sqrt(keep * (1.0 + p * _ALPHA_P * _ALPHA_P))
    b = -a * p * _ALPHA_P
    return a, b


def _channels_first_input(channels=512):
    torch.manual_seed(0)
    return torch.randn(4, channels, 2, 2)


def test_feature_alpha_dropout_eval_is_identity():
    x = _channels_first_input(64) * 3.0
    out = feature_alpha_dropout(x, 0.5, train=False)
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_feature_alpha_dropout_p1_zeros():
    x = _channels_first_input(64)
    out = feature_alpha_dropout(x, 1.0, train=True)
    assert bool((out == 0).all())


def test_feature_alpha_dropout_whole_channel_structure_and_rate():
    p = 0.4
    x = _channels_first_input(512)
    a, b = _affine(p)
    dropped_val = a * _ALPHA_P + b

    out = feature_alpha_dropout(x, p, train=True)

    n, c = out.shape[0], out.shape[1]
    dropped_channels = 0
    for i in range(n):
        for j in range(c):
            chan = out[i, j]
            chan_dropped = bool((chan - dropped_val).abs().max() < 1e-5)
            chan_kept = bool((chan - (a * x[i, j] + b)).abs().max() < 1e-5)
            assert chan_dropped or chan_kept, (
                "a channel is either fully dropped or fully kept"
            )
            dropped_channels += int(chan_dropped)
    rate = dropped_channels / (n * c)
    sigma = math.sqrt(p * (1 - p) / (n * c))
    assert abs(rate - p) < 4 * sigma + 0.05, f"channel drop rate {rate}"


def test_feature_alpha_dropout_empty():
    empty = torch.empty(0, 3, 2, 2)
    assert feature_alpha_dropout(empty, 0.5, train=True).shape == (0, 3, 2, 2)


def test_feature_alpha_dropout_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.feature_alpha_dropout")
    launches = []
    real_run = mod._feature_alpha_dropout_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._feature_alpha_dropout_kernel, "run", spy_run)
    x = torch.randn(2, 8, 4, 4)
    out = mod.feature_alpha_dropout(x, 0.5, train=True)
    assert len(launches) >= 1, "feature_alpha_dropout must launch the Triton kernel"
    a, b = _affine(0.5)
    dropped_val = a * _ALPHA_P + b
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            chan = out[i, j]
            chan_dropped = bool((chan - dropped_val).abs().max() < 1e-5)
            chan_kept = bool((chan - (a * x[i, j] + b)).abs().max() < 1e-5)
            assert chan_dropped or chan_kept
