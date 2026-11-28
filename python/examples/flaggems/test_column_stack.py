import importlib

import pytest
import torch

from .column_stack import column_stack, column_stack_out


def _spy_on(kernel, launches):
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    return spy_run


def test_column_stack_1d_columns_match_aten():
    a = torch.arange(3, dtype=torch.float64)
    b = torch.arange(3, dtype=torch.float64) * 10
    out = column_stack([a, b])
    ref = torch.column_stack([a, b])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (3, 2)


def test_column_stack_2d_and_3d_match_aten():
    a = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    b = torch.arange(2, dtype=torch.float64)
    out = column_stack([a, b])
    torch.testing.assert_close(out, torch.column_stack([a, b]), rtol=0, atol=0)
    t1 = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    t2 = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2)
    out3 = column_stack([t1, t2])
    ref3 = torch.column_stack([t1, t2])
    assert out3.shape == ref3.shape == (2, 5, 2)
    torch.testing.assert_close(out3, ref3, rtol=0, atol=0)


def test_column_stack_0d_inputs_match_aten():
    out = column_stack([torch.tensor(5.0), torch.tensor(6.0)])
    ref = torch.column_stack([torch.tensor(5.0), torch.tensor(6.0)])
    assert out.shape == (1, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_column_stack_dtype_promotion():
    a = torch.arange(3, dtype=torch.float32)
    b = torch.arange(3, dtype=torch.float64)
    out = column_stack([a, b])
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, torch.column_stack([a, b]), rtol=0, atol=0)
    c = torch.arange(3, dtype=torch.int64)
    out2 = column_stack([a, c])
    assert out2.dtype == torch.float32
    torch.testing.assert_close(out2, torch.column_stack([a, c]), rtol=0, atol=0)


def test_column_stack_bool_and_non_contiguous():
    a = torch.tensor([True, False, True])
    b = torch.tensor([False, False, True])
    out = column_stack([a, b])
    assert out.dtype == torch.bool
    torch.testing.assert_close(out, torch.column_stack([a, b]), rtol=0, atol=0)
    s = torch.arange(6, dtype=torch.float64)[::2]
    assert not s.is_contiguous()
    out2 = column_stack([s, torch.arange(3, dtype=torch.float64) * 10])
    ref2 = torch.column_stack([s, torch.arange(3, dtype=torch.float64) * 10])
    torch.testing.assert_close(out2, ref2, rtol=0, atol=0)


def test_column_stack_empty_rows():
    out = column_stack(
        [torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)]
    )
    ref = torch.column_stack(
        [torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)]
    )
    assert out.shape == ref.shape == (0, 2)


def test_column_stack_out_variant():
    a = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    b = torch.arange(2, dtype=torch.float64)
    out = torch.empty(2, 3, dtype=torch.float64)
    ret = column_stack_out([a, b], out=out)
    assert ret is out
    torch.testing.assert_close(out, torch.column_stack([a, b]), rtol=0, atol=0)
    small = torch.empty(1, 1, dtype=torch.float64)
    ret2 = column_stack_out([a, b], out=small)
    assert ret2 is small and small.shape == (2, 3)
    torch.testing.assert_close(small, torch.column_stack([a, b]), rtol=0, atol=0)


def test_column_stack_out_dtype_error():
    a = torch.arange(3, dtype=torch.float64)
    out = torch.empty(3, 1, dtype=torch.int64)  # float -> int is not castable
    with pytest.raises(TypeError, match="can't be cast to the desired output type"):
        column_stack_out([a], out=out)


def test_column_stack_errors_match_aten():
    with pytest.raises(
        RuntimeError, match="column_stack expects a non-empty TensorList"
    ):
        column_stack([])
    with pytest.raises(
        RuntimeError,
        match=r"Sizes of tensors must match except in dimension 1\. "
        r"Expected size 3 but got size 4",
    ):
        column_stack([torch.randn(3, 2), torch.randn(4, 2)])
    with pytest.raises(RuntimeError, match="same number of dimensions: got 2 and 3"):
        column_stack([torch.randn(2, 2), torch.randn(2, 2, 2)])


def test_column_stack_triton_path_tile_and_rows_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.column_stack")
    tile, rows = [], []
    monkeypatch.setattr(
        mod._copy_tile_kernel, "run", _spy_on(mod._copy_tile_kernel, tile)
    )
    monkeypatch.setattr(
        mod._copy_rows_kernel, "run", _spy_on(mod._copy_rows_kernel, rows)
    )
    # 32x64 part: row_len == block and small -> tile kernel
    a = torch.arange(32 * 64, dtype=torch.float64).reshape(32, 64)
    out = column_stack([a, a * 10])
    assert len(tile) == 2
    torch.testing.assert_close(out, torch.column_stack([a, a * 10]), rtol=0, atol=0)
    # 5x100 part: row_len=100 not a multiple of block 128 -> rows kernel
    b = torch.arange(5 * 100, dtype=torch.float64).reshape(5, 100)
    out2 = column_stack([b])
    assert len(rows) >= 1
    torch.testing.assert_close(out2, torch.column_stack([b]), rtol=0, atol=0)


def test_column_stack_triton_path_big_tile_kernel(monkeypatch):
    mod = importlib.import_module("flaggems.column_stack")
    launches = []
    monkeypatch.setattr(
        mod._copy_big_tile_kernel, "run", _spy_on(mod._copy_big_tile_kernel, launches)
    )
    a = torch.arange(64 * 1024, dtype=torch.float64).reshape(64, 1024)
    out = column_stack([a, a * 10])
    assert len(launches) == 2
    torch.testing.assert_close(out, torch.column_stack([a, a * 10]), rtol=0, atol=0)


def test_column_stack_triton_path_generic_kernel(monkeypatch):
    mod = importlib.import_module("flaggems.column_stack")
    launches = []
    monkeypatch.setattr(
        mod._copy_generic_kernel, "run", _spy_on(mod._copy_generic_kernel, launches)
    )
    a = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    b = torch.arange(6, dtype=torch.float64).reshape(2, 3).t()
    assert not b.is_contiguous()
    out = column_stack([a, b])
    assert len(launches) == 1
    torch.testing.assert_close(out, torch.column_stack([a, b]), rtol=0, atol=0)
