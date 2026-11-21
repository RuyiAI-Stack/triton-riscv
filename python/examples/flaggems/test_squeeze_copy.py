import importlib

import torch

from .squeeze_copy import squeeze_copy


def test_squeeze_copy_matches_aten():
    x = torch.arange(2 * 1 * 3 * 1 * 4, dtype=torch.float64).reshape(2, 1, 3, 1, 4)
    out = squeeze_copy(x)
    ref = torch.ops.aten.squeeze_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (2, 3, 4)


def test_squeeze_copy_all_ones_to_scalar():
    x = torch.tensor([[7.5]], dtype=torch.float32)
    out = squeeze_copy(x)
    assert out.shape == ()
    assert out.item() == 7.5


def test_squeeze_copy_no_unit_dims_is_plain_copy():
    x = torch.randn(3, 4)
    out = squeeze_copy(x)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    assert out is not x
    out[0, 0] = 99.0
    assert x[0, 0] != 99.0  # value copy, not a view


def test_squeeze_copy_strided_input():
    x = torch.randn(4, 6).t()  # (6, 4) non-contiguous
    assert not x.is_contiguous()
    out = squeeze_copy(x)
    ref = torch.ops.aten.squeeze_copy(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (6, 4)


def test_squeeze_copy_empty():
    x = torch.empty(1, 0, 1, 3)
    out = squeeze_copy(x)
    assert out.shape == (0, 3)
    assert out.numel() == 0


def test_squeeze_copy_integer_dtypes():
    for dt in (torch.int64, torch.int32):
        x = torch.arange(6, dtype=dt).reshape(2, 1, 3)
        out = squeeze_copy(x)
        torch.testing.assert_close(out, torch.ops.aten.squeeze_copy(x), rtol=0, atol=0)
    xb = (torch.arange(6) % 2 == 0).reshape(2, 1, 3)
    torch.testing.assert_close(
        squeeze_copy(xb), torch.ops.aten.squeeze_copy(xb), rtol=0, atol=0
    )


def test_squeeze_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.squeeze_copy")
    launches = []
    real_run = mod._squeeze_copy_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._squeeze_copy_kernel, "run", spy_run)
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    out = squeeze_copy(x)
    assert len(launches) >= 1, "squeeze_copy must launch the Triton kernel"
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
