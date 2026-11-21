import importlib

import pytest
import torch

from .fill_diagonal_ import fill_diagonal_


def test_fill_diagonal_basic_matches_aten():
    for shape in [(4, 4), (3, 5), (5, 3), (2, 2, 2), (3, 3, 3)]:
        x = torch.arange(
            torch.tensor(shape).prod().item(), dtype=torch.float64
        ).reshape(shape)
        ours = x.clone()
        ref = x.clone()
        ret = fill_diagonal_(ours, 7.0)
        ref.fill_diagonal_(7.0)
        assert ret is ours  # in-place, returns self
        torch.testing.assert_close(ours, ref, rtol=0, atol=0, msg=str(shape))


def test_fill_diagonal_wrap_matches_aten():
    for shape in [(7, 3), (10, 4), (9, 2)]:
        x = torch.arange(shape[0] * shape[1], dtype=torch.float64).reshape(shape)
        ours = x.clone()
        ref = x.clone()
        fill_diagonal_(ours, -1.0, wrap=True)
        ref.fill_diagonal_(-1.0, wrap=True)
        torch.testing.assert_close(ours, ref, rtol=0, atol=0, msg=str(shape))


def test_fill_diagonal_strided_input():
    base = torch.arange(6 * 4, dtype=torch.float64).reshape(6, 4)
    x = base.t()  # (4, 6) non-contiguous view
    ours = x.clone()
    ref = x.clone()
    fill_diagonal_(ours, 2.5)
    ref.fill_diagonal_(2.5)
    torch.testing.assert_close(ours, ref, rtol=0, atol=0)


def test_fill_diagonal_1d_raises():
    with pytest.raises(RuntimeError, match="dimensions must larger than 1"):
        fill_diagonal_(torch.arange(4), 1.0)


def test_fill_diagonal_unequal_higher_dims_raises():
    with pytest.raises(RuntimeError, match="equal length"):
        fill_diagonal_(torch.zeros(3, 3, 4), 1.0)


def test_fill_diagonal_integer_dtype():
    x = torch.zeros(4, 4, dtype=torch.int64)
    fill_diagonal_(x, 9)
    assert x.diagonal().tolist() == [9, 9, 9, 9]


def test_fill_diagonal_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fill_diagonal_")
    launches = []
    real_run = mod.fill_diagonal_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.fill_diagonal_kernel, "run", spy_run)
    x = torch.zeros(5, 5, dtype=torch.float64)
    fill_diagonal_(x, 3.0)
    assert len(launches) >= 1
    assert x.diagonal().tolist() == [3.0] * 5
    assert x[0, 1] == 0.0  # off-diagonal untouched
