import pytest
import torch

from .fp8_matmul import (
    fp8_matmul,
)


GROUP_SIZE = 128


def _make_fp8(x_f32):
    return x_f32.to(torch.float8_e4m3fn)


def _dequantize(a, a_s, b, b_s):
    k = a.shape[-1]
    a_scale = a_s.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=-1)[..., :k]
    a_deq = a.to(torch.float32) * a_scale

    if b_s.numel() == 0:
        b_scale = torch.ones_like(b, dtype=torch.float32)
    elif b_s.shape[0] == b.shape[0]:
        b_scale = b_s.to(torch.float32).repeat_interleave(GROUP_SIZE, dim=-1)[:, :k]
    else:
        b_scale = (
            b_s.to(torch.float32)[0]
            .repeat_interleave(GROUP_SIZE, dim=-1)
            .unsqueeze(0)
            .expand(b.shape[0], -1)[:, :k]
        )
    b_deq = b.to(torch.float32) * b_scale
    return a_deq, b_deq


@pytest.mark.parametrize("M, N, K", [(64, 64, 128), (128, 64, 256)])
def test_fp8_matmul_forward(M, N, K):
    torch.manual_seed(0)
    a_f32 = torch.randn(M, K, dtype=torch.float32)
    b_f32 = torch.randn(N, K, dtype=torch.float32)
    a = _make_fp8(a_f32)
    b = _make_fp8(b_f32)
    a_s = torch.ones(M, K // GROUP_SIZE, dtype=torch.float32)
    b_s = torch.ones(N // GROUP_SIZE, K // GROUP_SIZE, dtype=torch.float32)

    a_ref, b_ref = _dequantize(a, a_s, b, b_s)
    ref = torch.mm(a_ref, b_ref.t()).to(torch.bfloat16)
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

    a_ref, b_ref = _dequantize(a, a_s, b, b_s)
    ref = torch.matmul(a_ref, b_ref.t()).to(torch.bfloat16)
    tri_out = fp8_matmul(a, a_s, b, b_s)

    torch.testing.assert_close(tri_out, ref, rtol=1e-1, atol=1e-1)
