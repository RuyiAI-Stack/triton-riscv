import importlib

import pytest
import torch

from .row_stack import row_stack, row_stack_out


def _spy_on(kernel, launches):
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    return spy_run


def test_row_stack_1d_matches_aten():
    a = torch.arange(3, dtype=torch.float64)
    b = torch.arange(3, dtype=torch.float64) * 10
    out = row_stack([a, b])
    ref = torch.row_stack([a, b])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.shape == (2, 3)


def test_row_stack_0d_and_mixed_ranks_match_aten():
    out = row_stack([torch.tensor(5.0), torch.tensor(6.0)])
    ref = torch.row_stack([torch.tensor(5.0), torch.tensor(6.0)])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    a = torch.arange(3, dtype=torch.float64)
    m = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    out2 = row_stack([a, m])
    ref2 = torch.row_stack([a, m])
    assert out2.shape == (3, 3)
    torch.testing.assert_close(out2, ref2, rtol=0, atol=0)


def test_row_stack_2d_and_3d_match_aten():
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.ones(1, 3, dtype=torch.float64)
    out = row_stack([a, b])
    torch.testing.assert_close(out, torch.row_stack([a, b]), rtol=0, atol=0)
    t1 = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    t2 = torch.ones(2, 2, 2, dtype=torch.float64)
    out3 = row_stack([t1, t2])
    torch.testing.assert_close(out3, torch.row_stack([t1, t2]), rtol=0, atol=0)


def test_row_stack_dtype_promotion():
    a = torch.arange(3, dtype=torch.float32)
    b = torch.arange(3, dtype=torch.float64)
    out = row_stack([a, b])
    ref = torch.row_stack([a, b])
    assert out.dtype == ref.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    c = torch.arange(3, dtype=torch.int64)
    out2 = row_stack([a, c])
    assert out2.dtype == torch.float32
    torch.testing.assert_close(out2, torch.row_stack([a, c]), rtol=0, atol=0)


def test_row_stack_non_contiguous_and_bool():
    nc = torch.arange(6, dtype=torch.float64).reshape(2, 3).t()
    assert not nc.is_contiguous()
    out = row_stack([nc, torch.ones(3, 2, dtype=torch.float64)])
    ref = torch.row_stack([nc, torch.ones(3, 2, dtype=torch.float64)])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    bo = row_stack([torch.tensor([True, False]), torch.tensor([False, True])])
    ref_bo = torch.row_stack([torch.tensor([True, False]), torch.tensor([False, True])])
    assert bo.dtype == torch.bool
    torch.testing.assert_close(bo, ref_bo, rtol=0, atol=0)


def test_row_stack_channels_last_matches_aten():
    a = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    a = a.contiguous(memory_format=torch.channels_last)
    b = torch.ones(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
    assert not a.is_contiguous() and a.is_contiguous(memory_format=torch.channels_last)
    out = row_stack([a, b])
    ref = torch.row_stack([a, b])
    assert out.shape == ref.shape == (4, 3, 4, 5)
    assert out.stride() == ref.stride()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_row_stack_empty_rows():
    out = row_stack([torch.empty(0, 3), torch.ones(2, 3)])
    ref = torch.row_stack([torch.empty(0, 3), torch.ones(2, 3)])
    assert out.shape == ref.shape == (2, 3)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    all_empty = row_stack([torch.empty(0, 3), torch.empty(0, 3)])
    assert all_empty.shape == (0, 3)
    zero_width = row_stack([torch.empty(2, 0), torch.empty(3, 0)])
    ref_zw = torch.row_stack([torch.empty(2, 0), torch.empty(3, 0)])
    assert zero_width.shape == ref_zw.shape == (5, 0)


def test_row_stack_out_variant():
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.ones(1, 3, dtype=torch.float64)
    out = torch.empty(3, 3, dtype=torch.float64)
    ret = row_stack_out([a, b], out=out)
    assert ret is out
    torch.testing.assert_close(out, torch.row_stack([a, b]), rtol=0, atol=0)
    small = torch.empty(1, 3, dtype=torch.float64)
    ret2 = row_stack_out([a, b], out=small)
    assert ret2 is small and small.shape == (3, 3)
    torch.testing.assert_close(small, torch.row_stack([a, b]), rtol=0, atol=0)
    # non-contiguous out goes through the staging path, values still exact
    nc_out = torch.empty(9, dtype=torch.float64).as_strided((3, 3), (1, 3), 0)
    assert not nc_out.is_contiguous()
    ret3 = row_stack_out([a, b], out=nc_out)
    assert ret3 is nc_out
    torch.testing.assert_close(ret3, torch.row_stack([a, b]), rtol=0, atol=0)
    # a contiguous as_strided out takes the fast path directly
    strided_out = torch.empty(9, dtype=torch.float64).as_strided((3, 3), (3, 1), 0)
    ret4 = row_stack_out([a, b], out=strided_out)
    assert ret4 is strided_out
    torch.testing.assert_close(ret4, torch.row_stack([a, b]), rtol=0, atol=0)


def test_row_stack_errors_match_aten():
    with pytest.raises(RuntimeError, match="vstack expects a non-empty TensorList"):
        row_stack([])
    with pytest.raises(
        RuntimeError,
        match=r"Sizes of tensors must match except in dimension 0\. "
        r"Expected size 3 but got size 4",
    ):
        row_stack([torch.randn(2, 3), torch.randn(2, 4)])
    with pytest.raises(RuntimeError, match="same number of dimensions: got 2 and 3"):
        row_stack([torch.randn(2, 3), torch.randn(2, 2, 2)])
    with pytest.raises(TypeError, match="must be tuple of Tensors, not Tensor"):
        row_stack(torch.arange(3))


def test_row_stack_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.row_stack")
    launches = []
    monkeypatch.setattr(
        mod._row_stack_copy_kernel, "run", _spy_on(mod._row_stack_copy_kernel, launches)
    )
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.ones(1, 3, dtype=torch.float64)
    c = torch.ones(4, 3, dtype=torch.float64) * 5
    out = row_stack([a, b, c])
    assert len(launches) == 3  # one launch per input, any tensor count
    torch.testing.assert_close(out, torch.row_stack([a, b, c]), rtol=0, atol=0)
