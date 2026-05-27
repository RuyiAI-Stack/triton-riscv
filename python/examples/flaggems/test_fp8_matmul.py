import pytest
import torch

from .fp8_matmul import (
    fp8_matmul,
)


GROUP_SIZE = 128


def _make_fp8(x_f32):
    return x_f32.to(torch.float8_e4m3fn)


@pytest.mark.parametrize("M, N, K", [(64, 64, 128), (128, 64, 256)])
def test_fp8_matmul_forward(M, N, K):
    torch.manual_seed(0)
    a_f32 = torch.randn(M, K, dtype=torch.float32)
    b_f32 = torch.randn(N, K, dtype=torch.float32)
    a = _make_fp8(a_f32)
    b = _make_fp8(b_f32)
    a_s = torch.ones(M, K // GROUP_SIZE, dtype=torch.float32)
    b_s = torch.ones(N // GROUP_SIZE, K // GROUP_SIZE, dtype=torch.float32)

    ref = torch.mm(a_f32, b_f32.t()).to(torch.bfloat16)
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
    b_s = torch.ones(N // GROUP_SIZE, K // GROUP_SIZE, dtype=torch.float32)

    ref = torch.matmul(a_f32, b_f32.t()).to(torch.bfloat16)
    tri_out = fp8_matmul(a, a_s, b, b_s)

    torch.testing.assert_close(tri_out, ref, rtol=1e-1, atol=1e-1)
