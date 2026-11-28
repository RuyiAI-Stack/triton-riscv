import importlib

import pytest
import torch

from .slice_copy import slice_copy, slice_copy_out


def test_slice_copy_matches_aten():
    x = torch.arange(2 * 5 * 4, dtype=torch.float64).reshape(2, 5, 4)
    cases = [
        (0, 0, 1, 1),
        (1, 1, 4, 1),
        (1, 1, None, 2),
        (2, None, None, 3),
        (1, -3, None, 1),
        (1, 2, -1, 2),
        (-1, 1, 3, 1),
    ]
    for dim, start, end, step in cases:
        out = slice_copy(x, dim, start, end, step)
        ref = torch.ops.aten.slice_copy(x, dim, start, end, step)
        torch.testing.assert_close(
            out, ref, rtol=0, atol=0, msg=f"{dim},{start},{end},{step}"
        )
        assert out.shape == ref.shape


def test_slice_copy_negative_step_raises():
    x = torch.arange(8)
    with pytest.raises(RuntimeError, match="step must be positive"):
        slice_copy(x, 0, 0, 8, -1)


def test_slice_copy_zero_dim_raises():
    x = torch.tensor(1.0)
    with pytest.raises(IndexError, match="0-dim"):
        slice_copy(x, 0, 0, 1)


def test_slice_copy_dim_out_of_range():
    x = torch.randn(3, 4)
    with pytest.raises(IndexError, match="Dimension out of range"):
        slice_copy(x, 2, 0, 2)


def test_slice_copy_strided_input():
    x = torch.randn(4, 6).t()
    assert not x.is_contiguous()
    out = slice_copy(x, 1, 1, 5, 2)
    ref = torch.ops.aten.slice_copy(x, 1, 1, 5, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_slice_copy_complex_input():
    torch.manual_seed(0)
    x = torch.randn(3, 6, dtype=torch.complex128)
    out = slice_copy(x, 1, 1, None, 2)
    ref = torch.ops.aten.slice_copy(x, 1, 1, None, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_slice_copy_bool_input():
    x = (torch.arange(12) % 3 == 0).reshape(3, 4)
    out = slice_copy(x, 1, 0, None, 2)
    torch.testing.assert_close(
        out, torch.ops.aten.slice_copy(x, 1, 0, None, 2), rtol=0, atol=0
    )


def test_slice_copy_clamped_bounds():
    x = torch.arange(5, dtype=torch.float64)
    out = slice_copy(x, 0, 3, 99, 1)  # end clamps to 5
    torch.testing.assert_close(out, x[3:5], rtol=0, atol=0)
    out2 = slice_copy(x, 0, -99, 2, 1)  # start clamps to 0
    torch.testing.assert_close(out2, x[0:2], rtol=0, atol=0)


def test_slice_copy_empty_result():
    x = torch.arange(6, dtype=torch.float32)
    out = slice_copy(x, 0, 5, 3, 1)
    assert out.shape == (0,)


def test_slice_copy_out_variant():
    x = torch.arange(2 * 5 * 4, dtype=torch.float64).reshape(2, 5, 4)
    out = torch.empty(2, 2, 4, dtype=torch.float64)
    ret = slice_copy_out(x, 1, 1, 3, 1, out=out)
    assert ret is out
    torch.testing.assert_close(out, x[:, 1:3, :], rtol=0, atol=0)


def test_slice_copy_out_dtype_mismatch_raises():
    x = torch.arange(6, dtype=torch.float32)
    out = torch.empty(3, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="dtype float, but got double"):
        slice_copy_out(x, 0, 0, 3, 1, out=out)


def test_slice_copy_triton_path_all_three_kernels(monkeypatch):
    mod = importlib.import_module("flaggems.slice_copy")
    for kname in (
        "slice_copy_kernel_inner1",
        "slice_copy_kernel_step1",
        "slice_copy_kernel",
    ):
        launches = []
        kernel = getattr(mod, kname)
        real_run = kernel.run

        def spy_run(*args, **kwargs):
            launches.append(kwargs.get("grid"))
            return real_run(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy_run)
        x = torch.arange(2 * 5 * 4, dtype=torch.float64).reshape(2, 5, 4)
        if kname.endswith("inner1"):
            out = slice_copy(x, 2, 0, None, 2)  # trailing dim -> inner==1 kernel
            torch.testing.assert_close(out, x[:, :, ::2], rtol=0, atol=0)
        elif kname.endswith("step1"):
            out = slice_copy(x, 1, 1, 4, 1)  # step==1 with inner>1
            torch.testing.assert_close(out, x[:, 1:4, :], rtol=0, atol=0)
        else:
            out = slice_copy(x, 1, 0, None, 2)  # general: step>1, inner>1
            torch.testing.assert_close(out, x[:, ::2, :], rtol=0, atol=0)
        assert len(launches) >= 1, f"{kname} must be launched by its slice shape"
