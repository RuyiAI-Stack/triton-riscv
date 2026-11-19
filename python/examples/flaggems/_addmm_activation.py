import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# addmm with a fused activation: act(beta * bias + alpha * (mat1 @ mat2)).
# Probed on torch CPU: act=relu for use_gelu=False and the erf-form gelu for
# use_gelu=True; the bias broadcast is the general (M, N) broadcast of aten.
#
# The matmul is a correctness-first teaching kernel (one k-slice at a time,
# fp32 accumulator; fp64 inputs keep a native double accumulator).
# Non-fp64 inputs compute in fp32 (libdevice has no fp16/bf16 signatures).


@triton.jit
def _addmm_activation_kernel(
    bias_ptr,
    m1_ptr,
    m2_ptr,
    out_ptr,
    M,
    N,
    K,
    beta,
    alpha,
    stride_bm,
    stride_bn,
    stride_m1m,
    stride_m1k,
    stride_m2k,
    stride_m2n,
    stride_om,
    stride_on,
    USE_GELU: tl.constexpr,
    IS64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    mmask = offs_m < M
    nmask = offs_n < N

    if IS64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(K):
        a_col = tl.load(
            m1_ptr + offs_m * stride_m1m + kk * stride_m1k, mask=mmask, other=0.0
        )
        b_row = tl.load(
            m2_ptr + kk * stride_m2k + offs_n * stride_m2n, mask=nmask, other=0.0
        )
        if IS64:
            acc += a_col.to(tl.float64)[:, None] * b_row.to(tl.float64)[None, :]
        else:
            acc += a_col.to(tl.float32)[:, None] * b_row.to(tl.float32)[None, :]

    bias = tl.load(
        bias_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn,
        mask=mask,
        other=0.0,
    )
    if IS64:
        bias = bias.to(tl.float64)
    else:
        bias = bias.to(tl.float32)
    res = beta * bias + alpha * acc

    if USE_GELU:
        cdf = 0.5 * (1.0 + libdevice.erf(res * 0.7071067811865476))
        res = res * cdf
    else:
        res = tl.where(res > 0, res, 0.0)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        res.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _addmm_activation(bias, mat1, mat2, *, beta=1, alpha=1, use_gelu=False, out=None):
    if mat1.shape[1] != mat2.shape[0]:
        raise AssertionError("Incompatible dimensions")
    M, K = mat1.shape
    N = mat2.shape[1]
    try:
        torch.broadcast_shapes(bias.shape, (M, N))
    except RuntimeError as err:
        raise AssertionError("Incompatible input shape") from err
    bias_b = bias.expand(M, N).contiguous()
    m1 = mat1.contiguous()
    m2 = mat2.contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=m1.dtype, device=m1.device)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    _addmm_activation_kernel[grid](
        bias_b,
        m1,
        m2,
        out,
        M,
        N,
        K,
        float(beta),
        float(alpha),
        bias_b.stride(0),
        bias_b.stride(1),
        m1.stride(0),
        m1.stride(1),
        m2.stride(0),
        m2.stride(1),
        out.stride(0),
        out.stride(1),
        USE_GELU=bool(use_gelu),
        IS64=m1.dtype == torch.float64,
        BLOCK_M=16,
        BLOCK_N=16,
    )
    return out
