import importlib

import pytest
import torch

from .unbind_copy import unbind_copy


def test_unbind_copy_matches_aten():
    x = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5)
    for dim in [0, 1, 2, -1, -3]:
        outs = unbind_copy(x, dim)
        ref = torch.ops.aten.unbind_copy(x, dim)
        assert len(outs) == len(ref)
        for o, r in zip(outs, ref):
            torch.testing.assert_close(o, r, rtol=0, atol=0)


def test_unbind_copy_is_value_copy():
    x = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    outs = unbind_copy(x, 1)  # slices of shape (2,)
    outs[0][0] = -1.0
    assert x[0, 0] == 0.0


def test_unbind_copy_strided_input():
    x = torch.randn(4, 6).t()  # (6, 4) non-contiguous
    outs = unbind_copy(x, 0)
    ref = torch.ops.aten.unbind_copy(x, 0)
    for o, r in zip(outs, ref):
        torch.testing.assert_close(o, r, rtol=0, atol=0)


def test_unbind_copy_empty_dim():
    x = torch.empty(0, 3)
    assert unbind_copy(x, 0) == []


def test_unbind_copy_dim_out_of_range():
    x = torch.randn(3, 4)
    with pytest.raises(IndexError, match="Dimension out of range"):
        unbind_copy(x, 2)


def test_unbind_copy_1d_input():
    x = torch.arange(4, dtype=torch.float64)
    outs = unbind_copy(x, 0)
    assert len(outs) == 4
    for i, o in enumerate(outs):
        assert o.shape == () and o.item() == i


def test_unbind_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.unbind_copy")
    launches = []
    real_run = mod.unbind_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.unbind_copy_kernel, "run", spy_run)
    x = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5)
    outs = unbind_copy(x, 1)
    assert len(launches) >= 1
    for i, o in enumerate(outs):
        torch.testing.assert_close(o, x[:, i, :], rtol=0, atol=0)
    assert outs[0].shape == (3, 5)
