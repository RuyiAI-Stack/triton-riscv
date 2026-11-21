import importlib

import pytest
import torch

from .fliplr import fliplr


def test_fliplr_matches_aten():
    for shape in [(3, 5), (2, 4, 6), (1, 1), (17, 33)]:
        x = torch.arange(
            torch.tensor(shape).prod().item(), dtype=torch.float64
        ).reshape(shape)
        out = fliplr(x)
        torch.testing.assert_close(out, torch.fliplr(x), rtol=0, atol=0, msg=str(shape))


def test_fliplr_strided_input():
    x = torch.randn(4, 6).t()  # (6, 4) non-contiguous
    assert not x.is_contiguous()
    torch.testing.assert_close(fliplr(x), torch.fliplr(x), rtol=0, atol=0)


def test_fliplr_1d_raises():
    with pytest.raises(RuntimeError, match=">= 2-d"):
        fliplr(torch.arange(5))


def test_fliplr_empty():
    x = torch.empty(3, 0, 2)
    assert fliplr(x).shape == (3, 0, 2)


def test_fliplr_is_value_copy():
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    out = fliplr(x)
    out[0, 0] = -1.0
    assert x[0, 2] == 2.0


def test_fliplr_integer_dtypes():
    for dt in (torch.int64, torch.int32):
        x = torch.arange(2 * 4, dtype=dt).reshape(2, 4)
        torch.testing.assert_close(fliplr(x), torch.fliplr(x), rtol=0, atol=0)


def test_fliplr_contig_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fliplr")
    launches = []
    real_run = mod.fliplr_contiguous_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.fliplr_contiguous_kernel, "run", spy_run)
    x = torch.arange(2 * 5, dtype=torch.float64).reshape(2, 5)
    out = fliplr(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.flip(1), rtol=0, atol=0)


def test_fliplr_strided_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fliplr")
    launches = []
    real_run = mod.fliplr_strided_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.fliplr_strided_kernel, "run", spy_run)
    x = torch.randn(4, 6).t()
    out = fliplr(x)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.flip(1), rtol=0, atol=0)
