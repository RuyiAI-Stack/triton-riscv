import importlib

import pytest
import torch

from .transpose_copy import transpose_copy


def test_transpose_copy_matches_aten_2d():
    x = torch.arange(4 * 7, dtype=torch.float64).reshape(4, 7)
    out = transpose_copy(x, 0, 1)
    ref = torch.ops.aten.transpose_copy(x, 0, 1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (7, 4)


def test_transpose_copy_matches_aten_3d_4d():
    x3 = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    for d0, d1 in [(0, 1), (1, 2), (0, 2), (-1, -3)]:
        torch.testing.assert_close(
            transpose_copy(x3, d0, d1),
            torch.ops.aten.transpose_copy(x3, d0, d1),
            rtol=0,
            atol=0,
        )
    x4 = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
    for d0, d1 in [(0, 3), (1, 2), (2, 3)]:
        torch.testing.assert_close(
            transpose_copy(x4, d0, d1),
            torch.ops.aten.transpose_copy(x4, d0, d1),
            rtol=0,
            atol=0,
        )


def test_transpose_copy_tile_boundary_sizes():
    # Sizes straddling the 32x32 tile boundary catch mask bugs.
    for m, n in [(1, 1), (31, 33), (32, 32), (33, 31), (64, 65), (17, 100)]:
        x = torch.arange(m * n, dtype=torch.float64).reshape(m, n)
        torch.testing.assert_close(
            transpose_copy(x, 0, 1), x.t().contiguous(), rtol=0, atol=0, msg=f"{m}x{n}"
        )


def test_transpose_copy_strided_input():
    x = torch.randn(4, 6).t()  # (6, 4) non-contiguous
    out = transpose_copy(x, 0, 1)
    ref = torch.ops.aten.transpose_copy(x, 0, 1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_transpose_copy_1d_identity():
    x = torch.arange(7, dtype=torch.float64)
    out = transpose_copy(x, 0, 0)
    torch.testing.assert_close(out, x, rtol=0, atol=0)


def test_transpose_copy_same_dim_contiguous():
    x = torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3)
    out = transpose_copy(x, 1, 1)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    assert out is not x


def test_transpose_copy_complex_fallback():
    torch.manual_seed(0)
    x = torch.randn(3, 5, dtype=torch.complex128)
    out = transpose_copy(x, 0, 1)
    torch.testing.assert_close(
        out, torch.ops.aten.transpose_copy(x, 0, 1), rtol=0, atol=0
    )


def test_transpose_copy_rank5_fallback():
    x = torch.arange(2 * 2 * 2 * 2 * 2, dtype=torch.float64).reshape(2, 2, 2, 2, 2)
    out = transpose_copy(x, 0, 4)
    torch.testing.assert_close(
        out, torch.ops.aten.transpose_copy(x, 0, 4), rtol=0, atol=0
    )


def test_transpose_copy_dim_out_of_range():
    x = torch.randn(3, 4)
    with pytest.raises(IndexError, match="Dimension out of range"):
        transpose_copy(x, 0, 2)


def test_transpose_copy_empty():
    x = torch.empty(0, 5, dtype=torch.float32)
    out = transpose_copy(x, 0, 1)
    assert out.shape == (5, 0)


def test_transpose_copy_integer_dtypes():
    for dt in (torch.int64, torch.int32):
        x = torch.arange(4 * 4, dtype=dt).reshape(4, 4)
        torch.testing.assert_close(
            transpose_copy(x, 0, 1),
            torch.ops.aten.transpose_copy(x, 0, 1),
            rtol=0,
            atol=0,
        )


def test_transpose_copy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.transpose_copy")
    launches = []
    real_run = mod._transpose_copy_tiled_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._transpose_copy_tiled_kernel, "run", spy_run)
    x = torch.arange(4 * 7, dtype=torch.float64).reshape(4, 7)
    out = transpose_copy(x, 0, 1)
    assert len(launches) >= 1
    torch.testing.assert_close(out, x.t().contiguous(), rtol=0, atol=0)
    assert out.shape == (7, 4)
