import importlib

import pytest
import torch

from .channel_shuffle import channel_shuffle


def test_channel_shuffle_matches_aten():
    for shape, g in [
        ((2, 6, 3, 4), 2),
        ((1, 12, 2, 2), 3),
        ((3, 4, 2, 5), 4),
        ((2, 8, 3, 4), 8),
    ]:
        x = torch.arange(
            torch.tensor(shape).prod().item(), dtype=torch.float64
        ).reshape(shape)
        out = channel_shuffle(x, g)
        ref = torch.ops.aten.channel_shuffle(x, g)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"{shape},{g}")


def test_channel_shuffle_5d_input():
    # For >4D inputs upstream shuffles the trailing (C, H, W) block
    # (C = shape[-3]) while aten's dispatcher reads C from dim 1 -- the two
    # disagree, so the oracle here is the upstream semantics: shuffle the
    # flattened (N', C, H, W) view and reshape back.
    x = torch.arange(2 * 2 * 6 * 2 * 2, dtype=torch.float64).reshape(2, 2, 6, 2, 2)
    C, H, W = x.shape[-3:]
    ref = torch.ops.aten.channel_shuffle(x.reshape(-1, C, H, W), 3).reshape(x.shape)
    out = channel_shuffle(x, 3)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_channel_shuffle_2d_raises():
    with pytest.raises(ValueError, match="at least 3 dimensions"):
        channel_shuffle(torch.randn(4, 6), 2)


def test_channel_shuffle_bad_groups():
    x = torch.randn(2, 6, 3, 3)
    with pytest.raises(AssertionError, match="divisible"):
        channel_shuffle(x, 4)


def test_channel_shuffle_empty():
    x = torch.empty(2, 0, 3, 4)
    assert channel_shuffle(x, 2).shape == (2, 0, 3, 4)


def test_channel_shuffle_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.channel_shuffle")
    launches = []
    real_run = mod.channel_shuffle_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.channel_shuffle_kernel, "run", spy_run)
    x = torch.arange(2 * 6 * 2 * 2, dtype=torch.float64).reshape(2, 6, 2, 2)
    out = channel_shuffle(x, 3)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out, torch.ops.aten.channel_shuffle(x, 3), rtol=0, atol=0
    )
