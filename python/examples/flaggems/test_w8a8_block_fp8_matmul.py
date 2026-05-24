import torch
import triton

from .w8a8_block_fp8_matmul import w8a8_block_fp8_matmul


def test_w8a8_block_fp8_matmul_basic():
    torch.manual_seed(0)
    M, N, K = 4, 8, 16
    block_n, block_k = 8, 16

    A = torch.randn(M, K, dtype=torch.float32, device="cpu")
    B = torch.randn(N, K, dtype=torch.float32, device="cpu")

    As_shape = list(A.shape[:-1]) + [triton.cdiv(K, block_k)]
    As = torch.randn(As_shape, dtype=torch.float32, device="cpu")
    Bs_shape = [triton.cdiv(N, block_n), triton.cdiv(K, block_k)]
    Bs = torch.randn(Bs_shape, dtype=torch.float32, device="cpu")

    C = w8a8_block_fp8_matmul(
        A, B, As, Bs, block_size=[block_n, block_k], output_dtype=torch.float32
    )

    assert C.shape == (M, N)
    assert C.dtype == torch.float32


def test_w8a8_block_fp8_matmul_batched():
    torch.manual_seed(0)
    B, M, N, K = 3, 4, 8, 16
    block_n, block_k = 8, 16

    A = torch.randn(B, M, K, dtype=torch.float32, device="cpu")
    B_mat = torch.randn(N, K, dtype=torch.float32, device="cpu")

    As_shape = list(A.shape[:-1]) + [triton.cdiv(K, block_k)]
    As = torch.randn(As_shape, dtype=torch.float32, device="cpu")
    Bs_shape = [triton.cdiv(N, block_n), triton.cdiv(K, block_k)]
    Bs = torch.randn(Bs_shape, dtype=torch.float32, device="cpu")

    C = w8a8_block_fp8_matmul(
        A,
        B_mat,
        As,
        Bs,
        block_size=[block_n, block_k],
        output_dtype=torch.float32,
    )

    assert C.shape == (B, M, N)
    assert C.dtype == torch.float32
