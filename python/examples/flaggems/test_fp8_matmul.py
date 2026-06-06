import pytest
import torch

from .fp8_matmul import (
    fp8_matmul,
)


GROUP_SIZE = 128


def _make_fp8(x_f32):
    return x_f32.to(torch.float8_e4m3fn)


def _ceil_div(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("M, N, K", [(64, 64, 128), (128, 64, 256)])
def test_fp8_matmul_forward(M, N, K):
    torch.manual_seed(0)
    a_f32 = torch.randn(M, K, dtype=torch.float32)
    b_f32 = torch.randn(N, K, dtype=torch.float32)
    a = _make_fp8(a_f32)
    b = _make_fp8(b_f32)
    a_s = torch.ones(M, K // GROUP_SIZE, dtype=torch.float32)
    b_s = torch.ones(
        _ceil_div(N, GROUP_SIZE), K // GROUP_SIZE, dtype=torch.float32
    )

    ref = torch.mm(a.float(), b.float().t()).to(torch.bfloat16)
    tri_out = fp8_matmul(a, a_s, b, b_s)

    torch.testing.assert_close(tri_out, ref, rtol=1e-1, atol=1e-1)


def test_fp8_matmul_masks_tail_n_tile_with_ceil_scales():
    torch.manual_seed(0)
    M = 64
    N = 193
    K = 128
    a_f32 = torch.randn(M, K, dtype=torch.float32)
    b_f32 = torch.randn(N, K, dtype=torch.float32)
    a = _make_fp8(a_f32)
    b = _make_fp8(b_f32)
    a_s = torch.ones(M, K // GROUP_SIZE, dtype=torch.float32)
    b_s = torch.tensor([[1.0], [3.0]], dtype=torch.float32)

    n_scale = b_s[torch.arange(N) // GROUP_SIZE, 0]
    ref = (torch.mm(a.float(), b.float().t()) * n_scale).to(torch.bfloat16)
    tri_out = fp8_matmul(a, a_s, b, b_s)

    torch.testing.assert_close(tri_out, ref, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("shape", [(64, 128)])
def test_fp8_matmul_batched(shape):
    torch.manual_seed(0)
    B = 4
    M, K = shape
    N = 64
    a_f32 = torch.randn(B, M, K, dtype=torch.float32)
    b_f32 = torch.randn(N, K, dtype=torch.float32)
    a = _make_fp8(a_f32)
    b = _make_fp8(b_f32)
    a_s = torch.ones(B, M, K // GROUP_SIZE, dtype=torch.float32)
    b_s = torch.ones(
        _ceil_div(N, GROUP_SIZE), K // GROUP_SIZE, dtype=torch.float32
    )

    ref = torch.matmul(a.float(), b.float().t()).to(torch.bfloat16)
    tri_out = fp8_matmul(a, a_s, b, b_s)

    torch.testing.assert_close(tri_out, ref, rtol=1e-1, atol=1e-1)
