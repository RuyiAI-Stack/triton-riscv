"""Triton implementation of aten::ger (rank-1 outer product).

Ported from upstream FlagGems ops/ger.py.

out[i, j] = self[i] * vec2[j]

Call forms supported (matching the Gems pytest contract):
  ger(self, vec2)            -> newly allocated 2-D result
  ger(self, vec2, out=...)   -> writes into ``out`` and returns it unchanged
                                (also accepted positionally), per aten::ger.out.

Both operands must be 1-D; a rank violation raises ValueError before any
launch.  fp8 operands are rejected with RuntimeError, complex64 operands are
handled through real float32 views so no complex tensor is passed to the
Triton launcher.

Porting notes (repo conventions):
  - @libentry stripped (plain @triton.jit); no torch device context manager.
  - ``cache_modifier=".cs"`` on stores dropped: a CUDA streaming-cache hint
    with no effect on values; the CPU backend does not model it.
  - The autograd wrapper (_Ger and its backward kernels) is not ported: no
    test exercises requires_grad inputs (house precedent: addmm.py).
"""

import torch
import triton
import triton.language as tl

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)

_BLOCK_M = 64
_BLOCK_N = 64


@triton.jit
def _ger_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    M,
    N,
    sa,
    sb,
    so_m,
    so_n,
    NO_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SWIZZLE: tl.constexpr,
):
    # Column-major program order keeps consecutive CTAs writing adjacent memory.
    if SWIZZLE:
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_off = offs_m * so_m
    col_off = offs_n * so_n

    if NO_MASK:
        a = tl.load(a_ptr + offs_m * sa)
        b = tl.load(b_ptr + offs_n * sb)
        prod = a[:, None] * b[None, :]
        tl.store(o_ptr + row_off[:, None] + col_off[None, :], prod)
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        a = tl.load(a_ptr + offs_m * sa, mask=mask_m, other=0)
        b = tl.load(b_ptr + offs_n * sb, mask=mask_n, other=0)
        prod = a[:, None] * b[None, :]
        tl.store(
            o_ptr + row_off[:, None] + col_off[None, :],
            prod,
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _ger_complex64_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    M,
    N,
    sa,
    sb,
    so_m,
    so_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Complex outer product over float32 views: the pointer addresses the real
    (even) float component of each complex element; the imaginary component
    lives at +1."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    ar = tl.load(a_ptr + offs_m * sa, mask=mask_m, other=0.0)
    ai = tl.load(a_ptr + offs_m * sa + 1, mask=mask_m, other=0.0)
    br = tl.load(b_ptr + offs_n * sb, mask=mask_n, other=0.0)
    bi = tl.load(b_ptr + offs_n * sb + 1, mask=mask_n, other=0.0)

    # The products are forced to round to fp32 through an exact fp64 multiply
    # plus an explicit truncation, so the following fp32 add/sub cannot be
    # contracted into an FMA by the backend. aten's complex multiply is
    # unfused (round-per-operation); a fused FMA here differs by 1 ulp of the
    # products, which cancellation amplifies in the imaginary part.
    ar64 = ar.to(tl.float64)[:, None]
    ai64 = ai.to(tl.float64)[:, None]
    br64 = br.to(tl.float64)[None, :]
    bi64 = bi.to(tl.float64)[None, :]
    pr = (ar64 * br64).to(tl.float32)
    pi = (ai64 * bi64).to(tl.float32)
    qr = (ar64 * bi64).to(tl.float32)
    qi = (ai64 * br64).to(tl.float32)
    re = pr - pi
    im = qr + qi

    o_ptrs = o_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_n
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, re, mask=mask)
    tl.store(o_ptrs + 1, im, mask=mask)


def _forward_config(itemsize, M, N):
    """Shape-aware tile geometry for the real-dtype forward kernel."""
    if M * N >= (1 << 22):
        if itemsize <= 2:
            return 128, 128
        return 128, 64
    return _BLOCK_M, _BLOCK_N


def _launch(a, b, out):
    M = a.shape[0]
    N = b.shape[0]
    if out.is_complex():
        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
        av = torch.view_as_real(a)
        bv = torch.view_as_real(b)
        ov = torch.view_as_real(out)
        _ger_complex64_kernel[grid](
            av,
            bv,
            ov,
            M,
            N,
            av.stride(0),
            bv.stride(0),
            ov.stride(0),
            ov.stride(1),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
        )
    else:
        bm, bn = _forward_config(a.element_size(), M, N)
        no_mask = (M % bm == 0) and (N % bn == 0)
        swizzle = M * N >= (1 << 22)
        if swizzle:
            grid = (triton.cdiv(N, bn), triton.cdiv(M, bm))
        else:
            grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        _ger_kernel[grid](
            a,
            b,
            out,
            M,
            N,
            a.stride(0),
            b.stride(0),
            out.stride(0),
            out.stride(1),
            NO_MASK=no_mask,
            BLOCK_M=bm,
            BLOCK_N=bn,
            SWIZZLE=swizzle,
        )
    return out


def ger(self, vec2, out=None):
    if not isinstance(self, torch.Tensor) or not isinstance(vec2, torch.Tensor):
        raise TypeError("ger: expected Tensor arguments")
    if self.dim() != 1:
        raise ValueError("ger: Expected 1-D argument self")
    if vec2.dim() != 1:
        raise ValueError("ger: Expected 1-D argument vec2")
    if self.dtype in _FP8_DTYPES or vec2.dtype in _FP8_DTYPES:
        raise RuntimeError(
            "ger: fp8 operands are not supported by the NVIDIA backend "
            "(mul is not implemented for Float8)"
        )

    dtype = torch.result_type(self, vec2)
    M = self.shape[0]
    N = vec2.shape[0]

    if out is None:
        out = torch.empty((M, N), dtype=dtype, device=self.device)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("ger: out must be a Tensor")
        if out.dim() != 2 or out.shape[0] != M or out.shape[1] != N:
            raise ValueError("ger: out has an incompatible shape")
        if out.dtype != dtype:
            raise ValueError("ger: out has an incompatible dtype")

    if M == 0 or N == 0:
        return out

    return _launch(self.to(dtype), vec2.to(dtype), out)
