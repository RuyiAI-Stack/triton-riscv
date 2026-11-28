import importlib

import pytest
import torch

from .as_strided_copy import as_strided_copy, as_strided_copy_out


def test_as_strided_copy_matches_aten():
    x = torch.arange(4 * 6, dtype=torch.float64)
    cases = [
        ((2, 3), (6, 2), 1),
        ((3, 4), (4, 1), 0),
        ((2, 2), (8, 4), 5),
        ((24,), (1,), 0),
        ((2, 12), (12, 1), 0),
        ((6,), (4,), 3),
    ]
    for size, stride, offset in cases:
        out = as_strided_copy(x, size, stride, offset)
        ref = torch.ops.aten.as_strided_copy(x, size, stride, offset)
        torch.testing.assert_close(
            out, ref, rtol=0, atol=0, msg=f"{size},{stride},{offset}"
        )


def test_as_strided_copy_3d():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    out = as_strided_copy(x, (2, 2, 3), x.stride(), 0)
    ref = torch.ops.aten.as_strided_copy(x, (2, 2, 3), x.stride(), 0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_as_strided_copy_zero_dim():
    x = torch.tensor(3.5, dtype=torch.float64)
    out = as_strided_copy(x, (), ())
    assert out.shape == () and out.item() == 3.5


def test_as_strided_copy_none_offset():
    x = torch.arange(12, dtype=torch.float64)
    out = as_strided_copy(x, (3, 4), (4, 1))
    ref = torch.ops.aten.as_strided_copy(x, (3, 4), (4, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_as_strided_copy_empty():
    x = torch.arange(6, dtype=torch.float64)
    out = as_strided_copy(x, (0, 3), (3, 1), 0)
    assert out.shape == (0, 3)


def test_as_strided_copy_complex_fallback():
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.complex128)
    out = as_strided_copy(x, (2, 3), (4, 1), 1)
    ref = torch.ops.aten.as_strided_copy(x, (2, 3), (4, 1), 1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_as_strided_copy_out_variant():
    x = torch.arange(4 * 6, dtype=torch.float64)
    out = torch.empty(2, 3, dtype=torch.float64)
    ret = as_strided_copy_out(x, (2, 3), (6, 2), 1, out=out)
    assert ret is out
    torch.testing.assert_close(out, x.as_strided((2, 3), (6, 2), 1), rtol=0, atol=0)


def test_as_strided_copy_out_dtype_mismatch():
    x = torch.arange(6, dtype=torch.float32)
    out = torch.empty(3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        as_strided_copy_out(x, (3,), (1,), 0, out=out)


def test_as_strided_copy_tile_boundary():
    # 2d path uses 16x16 tiles; straddle the boundary.
    x = torch.arange(35 * 33, dtype=torch.float64)
    out = as_strided_copy(x, (35, 33), (33, 1), 0)
    torch.testing.assert_close(out, x.reshape(35, 33), rtol=0, atol=0)


def test_as_strided_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.as_strided_copy")
    launches = []
    real_run = mod._as_strided_copy_2d_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._as_strided_copy_2d_kernel, "run", spy_run)
    x = torch.arange(4 * 6, dtype=torch.float64)
    out = as_strided_copy(x, (2, 3), (6, 2), 1)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.as_strided((2, 3), (6, 2), 1), rtol=0, atol=0)
