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


def _activate_cpu_driver():
    if CPUDriver is None:
        raise RuntimeError(f"CPUDriver unavailable: {_CPU_DRIVER_IMPORT_ERROR}")
    triton.runtime.driver.set_active(CPUDriver())




@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_b = tl.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pid_x, pid_y
    else:
        grid_m = tl.num_programs(0)
        grid_n = tl.num_programs(1)
        pid = pid_x + pid_y * grid_m

        num_cta_per_group = grid_n * GROUP_M
        group_id = pid // num_cta_per_group
        inner_group_id = pid % num_cta_per_group
        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > grid_m, grid_m % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % group_size
        pid_n = inner_group_id // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    num_iters = tl.cdiv(K, TILE_K)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        mask_k = offs_k < K
        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)

        offs_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def bmm(a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("bmm expects torch.Tensor inputs")
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("bmm only supports CPU inputs in this example")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("bmm only supports torch.float32 inputs in this example")
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError("bmm expects 3D tensors shaped as (batch, m, k) and (batch, k, n)")
    if a.shape[0] != b.shape[0]:
        raise ValueError("Batch dim mismatch")
    if a.shape[2] != b.shape[1]:
        raise ValueError("K dim mismatch")

    _activate_cpu_driver()

    batch, M, K = a.shape
    _, _, N = b.shape
    out = torch.empty((batch, M, N), dtype=a.dtype, device=a.device)

    grid_fn = lambda meta: (
        triton.cdiv(M, meta["TILE_M"]),
        triton.cdiv(N, meta["TILE_N"]),
        batch,
    )
    bmm_kernel[grid_fn](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        TILE_M=32,
        TILE_N=32,
        TILE_K=32,
        GROUP_M=8,
    )
    return out


triton_bmm = bmm


def make_bmm_inputs(batch=2, m=8, n=10, k=6):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    a = torch.randn((batch, m, k), device="cpu", dtype=torch.float32, generator=generator)
    b = torch.randn((batch, k, n), device="cpu", dtype=torch.float32, generator=generator)
    return a, b


def bmm_reference(a, b):
    return torch.bmm(a, b)


def test_triton_bmm_matches_torch():
    a, b = make_bmm_inputs(batch=2, m=8, n=10, k=6)
    out = triton_bmm(a, b)
    ref = bmm_reference(a, b)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


def test_triton_bmm_tail_shape():
    a, b = make_bmm_inputs(batch=3, m=5, n=7, k=9)
    out = triton_bmm(a, b)
    ref = bmm_reference(a, b)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)
