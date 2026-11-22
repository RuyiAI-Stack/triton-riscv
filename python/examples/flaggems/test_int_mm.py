import importlib

import pytest
import torch

from .int_mm import int_mm, int_mm_out

# oracle: torch._int_mm runs on CPU and is exact int8 x int8 -> int32 (probed
# on torch 2.15.0+cpu, bit-identical to the int64 matmul reference).


def _int8(shape):
    return torch.randint(-128, 128, shape, dtype=torch.int8)


@pytest.mark.parametrize("M, N, K", [(3, 7, 5), (32, 32, 64), (17, 31, 33)])
def test_int_mm_matches_aten(M, N, K):
    torch.manual_seed(0)
    mat1 = _int8((M, K))
    mat2 = _int8((K, N))
    out = int_mm(mat1, mat2)
    ref = torch._int_mm(mat1, mat2)
    assert out.shape == (M, N) and out.dtype == torch.int32
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_int_mm_matches_int64_reference():
    torch.manual_seed(1)
    mat1 = _int8((65, 67))
    mat2 = _int8((67, 33))
    out = int_mm(mat1, mat2)
    ref = (mat1.to(torch.int64) @ mat2.to(torch.int64)).to(torch.int32)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_int_mm_extreme_values_exact():
    mat1 = torch.full((2, 4), -128, dtype=torch.int8)
    mat2 = torch.full((4, 3), 127, dtype=torch.int8)
    out = int_mm(mat1, mat2)
    expected = torch.full((2, 3), -128 * 127 * 4, dtype=torch.int32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_int_mm_k1():
    torch.manual_seed(2)
    mat1 = _int8((5, 1))
    mat2 = _int8((1, 7))
    out = int_mm(mat1, mat2)
    torch.testing.assert_close(out, mat1.int() @ mat2.int(), rtol=0, atol=0)


def test_int_mm_zero_contraction_dim_is_zero_output():
    mat1 = _int8((3, 0))
    mat2 = _int8((0, 5))
    out = int_mm(mat1, mat2)
    assert out.shape == (3, 5)
    assert torch.all(out == 0)


@pytest.mark.parametrize("empty_shape", [(0, 5, 3), (3, 0, 5)])
def test_int_mm_empty_outer_dims(empty_shape):
    torch.manual_seed(3)
    M, N, K = empty_shape
    mat1 = _int8((M, K))
    mat2 = _int8((K, N))
    out = int_mm(mat1, mat2)
    assert out.shape == (M, N)


def test_int_mm_transposed_layouts():
    torch.manual_seed(4)
    M, N, K = 9, 11, 13
    mat1 = _int8((K, M)).t()
    mat2 = _int8((N, K)).t()
    assert not mat1.is_contiguous() and not mat2.is_contiguous()
    out = int_mm(mat1, mat2)
    ref = torch._int_mm(mat1.contiguous(), mat2.contiguous())
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_int_mm_sliced_layouts():
    torch.manual_seed(5)
    M = 9
    mat1 = _int8((M, 26))[:, ::2]
    mat2 = _int8((26, 26))[::2, ::2]
    assert not mat1.is_contiguous()
    out = int_mm(mat1, mat2)
    torch.testing.assert_close(out, mat1.int() @ mat2.int(), rtol=0, atol=0)


def test_int_mm_expanded_layouts():
    torch.manual_seed(6)
    M, N, K = 5, 7, 9
    mat1 = _int8((1, K)).expand(M, K)
    mat2 = _int8((K, 1)).expand(K, N)
    out = int_mm(mat1, mat2)
    torch.testing.assert_close(out, mat1.int() @ mat2.int(), rtol=0, atol=0)


def test_int_mm_out_writes_buffer():
    torch.manual_seed(7)
    M, N, K = 3, 7, 5
    mat1 = _int8((M, K))
    mat2 = _int8((K, N))
    out = torch.empty((M, N), dtype=torch.int32)
    ret = int_mm_out(mat1, mat2, out=out)
    ref = torch._int_mm(mat1, mat2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert ret is out or torch.equal(ret, out)


def test_int_mm_out_non_contiguous_buffer():
    torch.manual_seed(8)
    M, N, K = 5, 3, 7
    mat1 = _int8((M, K))
    mat2 = _int8((K, N))
    buffer = torch.zeros((N, M), dtype=torch.int32)
    out = buffer.t()  # non-contiguous view; the copy-back path must fill it
    int_mm_out(mat1, mat2, out=out)
    ref = torch._int_mm(mat1, mat2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_int_mm_errors_match_upstream_messages():
    torch.manual_seed(9)
    mat1 = _int8((3, 5))
    mat2 = _int8((5, 7))
    with pytest.raises(RuntimeError, match="_int_mm: self must be a matrix"):
        int_mm(mat1[0], mat2)
    with pytest.raises(RuntimeError, match="_int_mm: mat2 must be a matrix"):
        int_mm(mat1, mat2[:, 0])
    with pytest.raises(
        RuntimeError, match="expected both inputs to have dtype torch.int8"
    ):
        int_mm(mat1.int(), mat2)
    with pytest.raises(RuntimeError, match="mat1 and mat2 shapes cannot be multiplied"):
        int_mm(mat1, _int8((4, 7)))
    with pytest.raises(RuntimeError, match="expected out to have dtype torch.int32"):
        int_mm_out(mat1, mat2, out=torch.empty((3, 7)))
    with pytest.raises(RuntimeError, match="expected out shape"):
        int_mm_out(mat1, mat2, out=torch.empty((3, 5), dtype=torch.int32))


def test_int_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.int_mm")
    launches = []
    kernel = mod._int_mm_dot_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    mat1 = _int8((9, 13))
    mat2 = _int8((13, 11))
    out = int_mm(mat1, mat2)
    ref = torch._int_mm(mat1, mat2)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
