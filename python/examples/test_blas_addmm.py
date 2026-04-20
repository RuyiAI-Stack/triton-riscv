import os
from pathlib import Path
from tempfile import mkdtemp

import pytest
import torch
import triton
import triton.language as tl

try:
    from triton.backends.triton_shared.driver import CPUDriver
except Exception as exc:  # pragma: no cover - collection guard
    CPUDriver = None
    _CPU_DRIVER_IMPORT_ERROR = exc
else:
    _CPU_DRIVER_IMPORT_ERROR = None


pytestmark = pytest.mark.skipif(
    CPUDriver is None, reason=f"CPUDriver unavailable: {_CPU_DRIVER_IMPORT_ERROR}"
)


@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    tl.store(c_ptrs, accumulator.to(bias.dtype), mask=c_mask)


def _activate_cpu_driver():
    if CPUDriver is None:
        raise RuntimeError(f"CPUDriver unavailable: {_CPU_DRIVER_IMPORT_ERROR}")
    triton.runtime.driver.set_active(CPUDriver())




def _validate_addmm_inputs(bias, mat1, mat2):
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias must be a torch.Tensor")
    if not isinstance(mat1, torch.Tensor):
        raise TypeError("mat1 must be a torch.Tensor")
    if not isinstance(mat2, torch.Tensor):
        raise TypeError("mat2 must be a torch.Tensor")

    for name, tensor in ("bias", bias), ("mat1", mat1), ("mat2", mat2):
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32")

    if mat1.dim() != 2:
        raise ValueError("mat1 must be rank-2")
    if mat2.dim() != 2:
        raise ValueError("mat2 must be rank-2")
    if mat1.shape[1] != mat2.shape[0]:
        raise ValueError("mat1.shape[1] must equal mat2.shape[0]")

    M, N = mat1.shape[0], mat2.shape[1]
    try:
        bias.broadcast_to((M, N))
    except RuntimeError as exc:
        raise ValueError("bias must be broadcastable to (M, N)") from exc


def addmm_reference(bias, mat1, mat2, *, beta=1.0, alpha=1.0):
    return torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)


def addmm(bias, mat1, mat2, *, beta=1.0, alpha=1.0):
    _validate_addmm_inputs(bias, mat1, mat2)
    _activate_cpu_driver()

    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    addmm_kernel[grid](
        mat1,
        mat2,
        bias,
        out,
        alpha,
        beta,
        M,
        N,
        K,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        bias.stride(0),
        bias.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    return out


triton_addmm = addmm


def make_addmm_inputs(m=15, n=19, k=17, *, matrix_bias=False, transposed_mat2=False):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    mat1 = torch.randn((m, k), device="cpu", dtype=torch.float32, generator=generator)
    if transposed_mat2:
        mat2 = torch.randn((n, k), device="cpu", dtype=torch.float32, generator=generator).t()
    else:
        mat2 = torch.randn((k, n), device="cpu", dtype=torch.float32, generator=generator)

    if matrix_bias:
        bias = torch.randn((m, n), device="cpu", dtype=torch.float32, generator=generator)
    else:
        bias = torch.randn((n,), device="cpu", dtype=torch.float32, generator=generator)
    return bias, mat1, mat2


def test_triton_addmm_vector_bias_matches_torch():
    bias, mat1, mat2 = make_addmm_inputs(m=15, n=19, k=17)
    out = triton_addmm(bias, mat1, mat2, beta=0.5, alpha=1.25)
    ref = addmm_reference(bias, mat1, mat2, beta=0.5, alpha=1.25)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


def test_triton_addmm_matrix_bias_transposed_mat2():
    bias, mat1, mat2 = make_addmm_inputs(
        m=7,
        n=10,
        k=13,
        matrix_bias=True,
        transposed_mat2=True,
    )
    assert not mat2.is_contiguous()
    out = triton_addmm(bias, mat1, mat2, beta=1.5, alpha=0.75)
    ref = addmm_reference(bias, mat1, mat2, beta=1.5, alpha=0.75)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


def test_triton_addmm_scalar_bias_broadcasts():
    bias = torch.tensor(1.5, device="cpu", dtype=torch.float32)
    mat1 = torch.randn((3, 4), device="cpu", dtype=torch.float32)
    mat2 = torch.randn((4, 5), device="cpu", dtype=torch.float32)

    out = triton_addmm(bias, mat1, mat2, beta=0.25, alpha=1.0)
    ref = addmm_reference(bias, mat1, mat2, beta=0.25, alpha=1.0)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


def test_triton_addmm_rejects_incompatible_mat_shapes():
    bias = torch.randn((5,), device="cpu", dtype=torch.float32)
    mat1 = torch.randn((4, 6), device="cpu", dtype=torch.float32)
    mat2 = torch.randn((7, 3), device="cpu", dtype=torch.float32)

    with pytest.raises(ValueError, match="mat1.shape\\[1\\] must equal mat2.shape\\[0\\]"):
        triton_addmm(bias, mat1, mat2)


def test_triton_addmm_rejects_non_broadcastable_bias():
    bias = torch.randn((3, 4), device="cpu", dtype=torch.float32)
    mat1 = torch.randn((2, 5), device="cpu", dtype=torch.float32)
    mat2 = torch.randn((5, 4), device="cpu", dtype=torch.float32)

    with pytest.raises(ValueError, match="bias must be broadcastable to \\(M, N\\)"):
        triton_addmm(bias, mat1, mat2)


def test_triton_addmm_rejects_rank_mismatch():
    bias = torch.randn((), device="cpu", dtype=torch.float32)
    mat1 = torch.randn((2, 3, 4), device="cpu", dtype=torch.float32)
    mat2 = torch.randn((4, 5), device="cpu", dtype=torch.float32)

    with pytest.raises(ValueError, match="mat1 must be rank-2"):
        triton_addmm(bias, mat1, mat2)


def test_triton_addmm_rejects_higher_rank_bias():
    bias = torch.randn((1, 3, 4), device="cpu", dtype=torch.float32)
    mat1 = torch.randn((3, 2), device="cpu", dtype=torch.float32)
    mat2 = torch.randn((2, 4), device="cpu", dtype=torch.float32)

    with pytest.raises(ValueError, match="bias must be broadcastable to \\(M, N\\)"):
        triton_addmm(bias, mat1, mat2)
