import importlib

import pytest
import torch

from .native_channel_shuffle import native_channel_shuffle


def test_native_channel_shuffle_matches_composition():
    # Documented semantic: (N, g, C//g, *spatial) -> transpose(1, 2).
    for shape, g in [
        ((2, 6, 3, 4), 2),
        ((1, 12, 2, 2), 3),
        ((3, 4, 5), 4),
        ((2, 8), 8),
        ((2, 6, 2, 3, 2), 3),
    ]:
        x = torch.arange(
            torch.tensor(shape).prod().item(), dtype=torch.float64
        ).reshape(shape)
        n, c, *spatial = shape
        ref = x.reshape(n, g, c // g, *spatial).swapdims(1, 2).reshape(x.shape)
        out = native_channel_shuffle(x, g)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"{shape},{g}")


def test_native_channel_shuffle_1d_raises():
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        native_channel_shuffle(torch.randn(6), 2)


def test_native_channel_shuffle_bad_groups():
    x = torch.randn(2, 6, 3)
    with pytest.raises(AssertionError, match="divisible"):
        native_channel_shuffle(x, 4)


def test_native_channel_shuffle_empty():
    x = torch.empty(2, 4, 0)
    assert native_channel_shuffle(x, 2).shape == (2, 4, 0)


def test_native_channel_shuffle_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.native_channel_shuffle")
    launches = []
    real_run = mod._native_channel_shuffle_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._native_channel_shuffle_kernel, "run", spy_run)
    x = torch.arange(2 * 6 * 2 * 2, dtype=torch.float64).reshape(2, 6, 2, 2)
    out = native_channel_shuffle(x, 3)
    assert len(launches) >= 1
    ref = x.reshape(2, 3, 2, 2, 2).swapdims(1, 2).reshape(x.shape)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
