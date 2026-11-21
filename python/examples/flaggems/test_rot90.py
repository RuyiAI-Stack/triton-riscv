import importlib

import pytest
import torch

from .rot90 import rot90


def test_rot90_2d_matches_aten():
    y = torch.arange(3 * 4, dtype=torch.float64).reshape(3, 4)
    for k in (0, 1, 2, 3):
        out = rot90(y, k)
        ref = torch.rot90(y, k)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"k={k}")
        assert out.shape == ref.shape


def test_rot90_3d_default_dims_matches_aten():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    for k in (0, 1, 2, 3):
        out = rot90(x, k)
        ref = torch.rot90(x, k)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"k={k}")
        assert out.shape == ref.shape


def test_rot90_other_dim_pairs_matches_aten():
    x = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    for dims in ([1, 2], [2, 0], [0, 2], [1, 0], [-2, -1], [-1, -2], [2, -3]):
        for k in (1, 2, 3):
            out = rot90(x, k, dims=list(dims))
            ref = torch.rot90(x, k, dims=list(dims))
            torch.testing.assert_close(
                out, ref, rtol=0, atol=0, msg=f"dims={dims},k={k}"
            )
            assert out.shape == ref.shape


def test_rot90_4d_matches_aten():
    x = torch.arange(2 * 2 * 2 * 2, dtype=torch.float64).reshape(2, 2, 2, 2)
    out = rot90(x, 1, dims=[2, 3])
    ref = torch.rot90(x, 1, dims=[2, 3])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_rot90_k_normalized_like_aten():
    y = torch.arange(3 * 4).reshape(3, 4)
    for k in (-1, -3, 5, 6, 8):
        out = rot90(y, k)
        ref = torch.rot90(y, k)  # aten also normalizes k % 4
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"k={k}")


def test_rot90_non_contiguous_input():
    torch.manual_seed(0)
    x = torch.randn(4, 3).t()
    assert not x.is_contiguous()
    out = rot90(x, 1)
    ref = torch.rot90(x, 1)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_rot90_other_dtypes():
    torch.manual_seed(1)
    for dtype in (torch.float16, torch.float32, torch.int64):
        x = torch.arange(2 * 5, dtype=dtype).reshape(2, 5)
        out = rot90(x, 1)
        torch.testing.assert_close(out, torch.rot90(x, 1), rtol=0, atol=0)


def test_rot90_errors_match_aten():
    x = torch.arange(6).reshape(2, 3)
    with pytest.raises(RuntimeError, match="expected total rotation dims == 2"):
        rot90(x, 1, dims=[0])
    with pytest.raises(
        RuntimeError, match="expected total dims >= 2, but got total dims = 1"
    ):
        rot90(torch.arange(4), 1)
    with pytest.raises(RuntimeError, match="expected rotation dims to be different"):
        rot90(x, 1, dims=[0, 0])
    with pytest.raises(IndexError, match="Dimension out of range"):
        rot90(x, 1, dims=[0, 5])


def test_rot90_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.rot90")
    launches = []
    kernel = mod.rot90_kernel_2d
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    y = torch.arange(3 * 4, dtype=torch.float64).reshape(3, 4)
    out = rot90(y, 1)
    assert len(launches) >= 1
    torch.testing.assert_close(out, torch.rot90(y, 1), rtol=0, atol=0)
