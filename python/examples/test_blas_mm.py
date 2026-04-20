import os

import pytest
import torch
import triton
import triton.language as tl
from pathlib import Path
from tempfile import mkdtemp

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


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@triton.jit
def mm_kernel_general(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
        mask=mask_k[None, :],
        other=0.0,
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
        mask=mask_k[:, None],
        other=0.0,
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


_ORDERED_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ORDERED_DTYPES
    assert b in _ORDERED_DTYPES

    for dtype in _ORDERED_DTYPES:
        if a is dtype:
            return b
        if b is dtype:
            return a


def _activate_cpu_driver():
    if CPUDriver is None:
        raise RuntimeError(f"CPUDriver unavailable: {_CPU_DRIVER_IMPORT_ERROR}")
    triton.runtime.driver.set_active(CPUDriver())




def _validate_mm_inputs(a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("mm expects torch.Tensor inputs")
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("mm only supports CPU inputs in this example")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("mm only supports torch.float32 inputs in this example")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("mm expects 2D tensors shaped as (m, k) and (k, n)")
    if a.shape[1] != b.shape[0]:
        raise ValueError("mm expects a.shape[1] == b.shape[0]")


def mm(a, b):
    _validate_mm_inputs(a, b)
    _activate_cpu_driver()
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    mm_kernel_general[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
        GROUP_M=8,
    )
    return c


triton_mm = mm


def make_mm_inputs(m=15, n=19, k=17, *, noncontiguous=False):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    if noncontiguous:
        a = torch.randn(
            (m * 2, k * 2),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        )[::2, ::2]
        b = torch.randn(
            (k * 2, n * 2),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        )[::2, ::2]
    else:
        a = torch.randn((m, k), device="cpu", dtype=torch.float32, generator=generator)
        b = torch.randn((k, n), device="cpu", dtype=torch.float32, generator=generator)
    return a, b


def mm_reference(a, b):
    return torch.mm(a, b)


def test_triton_mm_matches_torch():
    a, b = make_mm_inputs(m=15, n=19, k=17)
    out = triton_mm(a, b)
    ref = mm_reference(a, b)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


def test_triton_mm_noncontiguous_inputs():
    a, b = make_mm_inputs(m=15, n=19, k=17, noncontiguous=True)
    assert not a.is_contiguous()
    assert not b.is_contiguous()
    out = triton_mm(a, b)
    ref = mm_reference(a, b)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


def test_triton_mm_tail_shape():
    a, b = make_mm_inputs(m=7, n=10, k=13)
    out = triton_mm(a, b)
    ref = mm_reference(a, b)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=0)


def test_triton_mm_rejects_non_tensor_inputs():
    b = torch.randn((4, 5), device="cpu", dtype=torch.float32)

    with pytest.raises(TypeError, match="mm expects torch.Tensor inputs"):
        triton_mm("not-a-tensor", b)


def test_triton_mm_rejects_rank_mismatch():
    a = torch.randn((2, 3, 4), device="cpu", dtype=torch.float32)
    b = torch.randn((4, 5), device="cpu", dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match="mm expects 2D tensors shaped as \\(m, k\\) and \\(k, n\\)",
    ):
        triton_mm(a, b)
