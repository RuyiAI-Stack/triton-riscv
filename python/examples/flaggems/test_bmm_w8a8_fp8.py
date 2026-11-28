import importlib

import pytest
import torch

from .bmm_w8a8_fp8 import bmm_w8a8_fp8

# fp8 operands with per-128-block scales and bf16 outputs: the output
# cast rounds at ~4e-3 relative and block-scale reordering adds slack;
# atol 1e-2 covers both (the upstream suite uses rtol 0.01 for this op).

# oracle: formula per upstream bmm_w8a8_fp8.py (no aten op): fp8 operands are
# decoded exactly and dotted in fp32; the accumulator multiplies each k-block
# partial by that block's A/B scales (kernel lines 95-134, block granularity
# from block_size=(128, 128, 128)), so the reference decomposes the
# contraction over 128-wide k blocks:
#   out[b, m, n] = sum_kb (A_bk @ B_bk) * a_s[b, m//128, kb] * b_s[b, kb, n//128]


def _fp8(shape, dtype=torch.float8_e4m3fn):
    return (torch.randn(shape) * 0.25).to(dtype)


def _ref(A, B, A_scale, B_scale, block_k=128):
    batch, M, K = A.shape
    N = B.shape[2]
    num_kb = -(-K // block_k)
    num_mb = -(-M // 128)
    num_nb = -(-N // 128)
    assert tuple(A_scale.shape) == (batch, num_mb, num_kb)
    assert tuple(B_scale.shape) == (batch, num_kb, num_nb)
    out = torch.zeros((batch, M, N), dtype=torch.float32)
    for kb in range(num_kb):
        k0, k1 = kb * block_k, min(K, (kb + 1) * block_k)
        partial = torch.bmm(A[:, :, k0:k1].float(), B[:, k0:k1].float())
        sa = A_scale[:, :, kb].float()  # (batch, num_mb)
        sb = B_scale[:, kb, :].float()  # (batch, num_nb)
        row = torch.arange(M) // 128
        col = torch.arange(N) // 128
        # out[b, m, n] += partial * a_s[b, m // 128, kb] * b_s[b, kb, n // 128]
        out += partial * sa[:, row][:, :, None] * sb[:, col][:, None, :]
    return out


@pytest.mark.parametrize(
    "batch, M, K, N",
    [(1, 128, 128, 128), (2, 256, 256, 256), (2, 256, 128, 384)],
)
def test_bmm_w8a8_fp8_block_aligned_matches_formula(batch, M, K, N):
    torch.manual_seed(0)
    A = _fp8((batch, M, K))
    B = _fp8((batch, K, N))
    A_scale = torch.rand(batch, M // 128, K // 128) + 0.5
    B_scale = torch.rand(batch, K // 128, N // 128) + 0.5

    out = bmm_w8a8_fp8(A, B, A_scale, B_scale)
    ref = _ref(A, B, A_scale, B_scale).to(torch.bfloat16)
    assert out.shape == (batch, M, N) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_bmm_w8a8_fp8_tail_blocks():
    # M < 128, K not a multiple of 128 and N odd: launch blocks collapse to
    # the problem size and the per-block scale indexing must follow cdiv.
    torch.manual_seed(1)
    batch, M, K, N = 1, 64, 200, 96
    A = _fp8((batch, M, K))
    B = _fp8((batch, K, N))
    num_mb, num_kb, num_nb = 1, 2, 1
    A_scale = torch.rand(batch, num_mb, num_kb) + 0.5
    B_scale = torch.rand(batch, num_kb, num_nb) + 0.5

    out = bmm_w8a8_fp8(A, B, A_scale, B_scale)
    ref = _ref(A, B, A_scale, B_scale).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_bmm_w8a8_fp8_single_k_block():
    # K < block_k (and K == 64, the size the wrapper's launch asserts accept
    # for a whole single tile): BLOCK_K == K path scales once after the
    # accumulation loop.
    torch.manual_seed(2)
    batch, M, K, N = 2, 32, 64, 128
    A = _fp8((batch, M, K))
    B = _fp8((batch, K, N))
    A_scale = torch.rand(batch, 1, 1) + 0.5
    B_scale = torch.rand(batch, 1, 1) + 0.5

    out = bmm_w8a8_fp8(A, B, A_scale, B_scale)
    ref = _ref(A, B, A_scale, B_scale).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_bmm_w8a8_fp8_zero_scales():
    torch.manual_seed(3)
    batch, M, K, N = 1, 128, 128, 128
    A = _fp8((batch, M, K))
    B = _fp8((batch, K, N))
    out = bmm_w8a8_fp8(A, B, torch.zeros(batch, 1, 1), torch.ones(batch, 1, 1))
    assert torch.all(out == 0)


def test_bmm_w8a8_fp8_e5m2_operands():
    torch.manual_seed(4)
    batch, M, K, N = 1, 128, 128, 128
    A = _fp8((batch, M, K), dtype=torch.float8_e5m2)
    B = _fp8((batch, K, N), dtype=torch.float8_e5m2)
    A_scale = torch.rand(batch, 1, 1) + 0.5
    B_scale = torch.rand(batch, 1, 1) + 0.5
    out = bmm_w8a8_fp8(A, B, A_scale, B_scale)
    ref = _ref(A, B, A_scale, B_scale).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_bmm_w8a8_fp8_out_dtype_and_buffer():
    torch.manual_seed(5)
    batch, M, K, N = 1, 128, 128, 128
    A = _fp8((batch, M, K))
    B = _fp8((batch, K, N))
    A_scale = torch.rand(batch, 1, 1) + 0.5
    B_scale = torch.rand(batch, 1, 1) + 0.5
    out = torch.empty((batch, M, N), dtype=torch.float16)
    ret = bmm_w8a8_fp8(A, B, A_scale, B_scale, out_dtype=torch.float16, out=out)
    ref = _ref(A, B, A_scale, B_scale).to(torch.float16)
    assert ret is out or torch.equal(ret, out)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_bmm_w8a8_fp8_empty_batch():
    torch.manual_seed(6)
    A = _fp8((0, 128, 128))
    B = _fp8((0, 128, 64))
    A_scale = torch.rand(0, 1, 1)
    B_scale = torch.rand(0, 1, 1)
    out = bmm_w8a8_fp8(A, B, A_scale, B_scale)
    assert out.shape == (0, 128, 64)


def test_bmm_w8a8_fp8_errors_are_assertions():
    torch.manual_seed(7)
    A = _fp8((2, 128, 128))
    B = _fp8((2, 128, 64))
    A_scale = torch.rand(2, 1, 1)
    B_scale = torch.rand(2, 1, 1)
    with pytest.raises(AssertionError):
        bmm_w8a8_fp8(A, B[:1], A_scale, B_scale)
    with pytest.raises(AssertionError):
        bmm_w8a8_fp8(A, B, A_scale, B_scale, block_size=(64, 64, 64))
    with pytest.raises(AssertionError):
        bmm_w8a8_fp8(A.float(), B, A_scale, B_scale)
    with pytest.raises(AssertionError):
        bmm_w8a8_fp8(A, B, A_scale, B_scale, out=torch.empty((2, 128, 63)))


def test_bmm_w8a8_fp8_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.bmm_w8a8_fp8")
    launches = []
    kernel = mod.bmm_w8a8_fp8_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    batch, M, K, N = 2, 256, 256, 256
    A = _fp8((batch, M, K))
    B = _fp8((batch, K, N))
    A_scale = torch.rand(batch, 2, 2) + 0.5
    B_scale = torch.rand(batch, 2, 2) + 0.5
    out = bmm_w8a8_fp8(A, B, A_scale, B_scale)
    ref = _ref(A, B, A_scale, B_scale).to(torch.bfloat16)
    assert len(launches) >= 1
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
