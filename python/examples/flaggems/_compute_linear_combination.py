"""_compute_linear_combination - batched weighted sum over leading rows.

Reference semantics (aten / flag_gems; callable on this build as
torch._compute_linear_combination on the CPU):
  input: [n, ...] tensor; coefficients: [m, n] tensor
  output[i, ...] = sum_j coefficients[i, j] * input[j, ...]
  i.e. a batched matrix-vector product ``coefficients @ input.flatten(1)``
  with the output shaped [m, *input.shape[1:]].
  With ``out`` the aten .out variant ACCUMULATES into the existing buffer
  (probed); the port mirrors that. An empty input raises aten's
  "Empty tensor not supported" RuntimeError; the upstream module asserts
  instead.

Port notes (triton-riscv, vs the upstream flag_gems file):
- Only upstream's generic outer-product kernel is ported. The tl.dot
  fast path is a matrix-unit optimization for GPU backends; this CPU
  backend has no such units and the try/except fallback machinery buys
  nothing here.
- The kernel computes ``out_2d (M, P) = Coeff (M, K) @ in_2d (K, P)`` with
  fp32 accumulation and casts to the output dtype on store; the trailing
  dimensions are addressed through their own strides (no copy) via the
  per-lane ``in_off_p`` decomposition (upstream verbatim).
- aten's CPU kernel silently misreads non-contiguous coefficients and
  tolerates a coefficients/input contraction mismatch (both probed); the
  port walks the coefficient strides and keeps upstream's shape assert.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["_compute_linear_combination", "_compute_linear_combination_out"]


@triton.jit(do_not_specialize=["M", "K", "P"])
def _compute_linear_combination_kernel(
    In,  # input, viewed as (n0, P) through its strides
    Coeff,  # coefficients, (m, n)
    Out,  # output, contiguous (m, P)
    M,
    K,  # contraction length
    P,
    IN_S0: tl.constexpr,  # stride of input dim 0
    COEFF_SM: tl.constexpr,
    COEFF_SK: tl.constexpr,
    T1D: tl.constexpr,
    T1S: tl.constexpr,
    T2D: tl.constexpr,
    T2S: tl.constexpr,
    T3D: tl.constexpr,
    T3S: tl.constexpr,
    T4S: tl.constexpr,
    ACCUM: tl.constexpr,  # add into an existing ``out`` buffer
    NTAIL: tl.constexpr,  # number of (extent, stride) slots describing the tail
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    m_mask = offs_m < M
    p_mask = offs_p < P

    # Offset of flat tail index ``p`` inside ``input``.
    if NTAIL == 1:
        in_off_p = offs_p * T1S
    elif NTAIL == 2:
        r1 = offs_p // T1D
        in_off_p = (offs_p % T1D) * T1S + r1 * T2S
    elif NTAIL == 3:
        r1 = offs_p // T1D
        r2 = r1 // T2D
        in_off_p = (offs_p % T1D) * T1S + (r1 % T2D) * T2S + r2 * T3S
    else:
        r1 = offs_p // T1D
        r2 = r1 // T2D
        r3 = r2 // T3D
        in_off_p = (offs_p % T1D) * T1S + (r1 % T2D) * T2S + (r2 % T3D) * T3S + r3 * T4S

    coeff_base = Coeff + offs_m * COEFF_SM
    in_base = In + in_off_p

    acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        for kk in tl.static_range(0, BLOCK_K):
            k = k0 + kk
            k_mask = k < K
            av = tl.load(
                coeff_base + kk * COEFF_SK, mask=m_mask & k_mask, other=0.0
            ).to(tl.float32)
            bv = tl.load(in_base + kk * IN_S0, mask=p_mask & k_mask, other=0.0).to(
                tl.float32
            )
            acc += av[:, None] * bv[None, :]
        coeff_base += BLOCK_K * COEFF_SK
        in_base += BLOCK_K * IN_S0

    # The output is always contiguous, i.e. flat (M, P) with row stride P.
    out_ptrs = Out + offs_m[:, None] * P + offs_p[None, :]
    store_mask = m_mask[:, None] & p_mask[None, :]
    if ACCUM:
        acc += tl.load(out_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=store_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _tail_slots(tensor):
    """Reduce dims >= 1 of ``tensor`` to (extent, stride) slots.

    Adjacent dims whose strides are compatible are merged, so a contiguous
    tail collapses to a single slot.
    """
    extents = []
    strides = []
    for d in range(1, tensor.dim()):
        size = tensor.shape[d]
        stride = tensor.stride(d)
        if size == 1:
            continue
        if extents and strides[-1] == stride * size:
            extents[-1] *= size
            strides[-1] = stride
        else:
            extents.append(size)
            strides.append(stride)
    if not extents:
        return [1], [0]
    return extents, strides


def _compute_linear_combination(input, coefficients, *, out=None):
    logger.debug("GEMS _COMPUTE_LINEAR_COMBINATION")

    # aten raises RuntimeError("Empty tensor not supported") here; the
    # upstream module asserts. The port mirrors aten's contract.
    if input.ndimension() <= 0 or input.numel() <= 0:
        raise RuntimeError("Empty tensor not supported")

    # coefficients is [m, n]; input is [n, ...]
    assert coefficients.dim() == 2, "coefficients must be 2-dimensional"
    m, n = coefficients.shape
    assert input.shape[0] == n, "incompatible dimensions: input and coefficients"

    output_shape = (m,) + tuple(input.shape[1:])

    if out is None:
        out = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        accum = False
        write_out = out
    else:
        assert tuple(out.shape) == output_shape, "Incompatible output shape"
        accum = True
        # The kernel writes a contiguous (m, P) buffer; stage through a
        # temporary for a non-contiguous ``out`` and copy back at the end.
        if out.is_contiguous():
            write_out = out
        else:
            write_out = torch.empty(output_shape, device=out.device, dtype=out.dtype)
            write_out.copy_(out)

    N = 1
    for dim in input.shape[1:]:
        N *= dim
    if m == 0 or N == 0:
        if write_out is not out:
            out.copy_(write_out)
        return out

    extents, strides = _tail_slots(input)
    if len(extents) > 4:
        raise RuntimeError("unsupported non-contiguous input layout")
    slots = [(extents[i], strides[i]) for i in range(len(extents) - 1, -1, -1)]
    while len(slots) < 4:
        slots.append((1, 0))
    # The kernel peels the innermost tail dimension first, so the slots are
    # supplied most-rapidly-varying-first.
    ntail = len(extents)

    block_m = min(_next_pow2(m), 64)
    block_p = min(_next_pow2(N), 256)
    while block_m * block_p > 4096:
        if block_p >= 64:
            block_p //= 2
        elif block_m > 1:
            block_m //= 2
        else:
            break
    block_k = 16

    tile = block_m * block_p
    if tile <= 256:
        num_warps = 1
    elif tile <= 1024:
        num_warps = 2
    else:
        num_warps = 4

    grid = (triton.cdiv(m, block_m), triton.cdiv(N, block_p))
    _compute_linear_combination_kernel[grid](
        input,
        coefficients,
        write_out,
        m,
        n,
        N,
        IN_S0=input.stride(0),
        COEFF_SM=coefficients.stride(0),
        COEFF_SK=coefficients.stride(1),
        T1D=slots[0][0],
        T1S=slots[0][1],
        T2D=slots[1][0],
        T2S=slots[1][1],
        T3D=slots[2][0],
        T3S=slots[2][1],
        T4S=slots[3][1],
        ACCUM=accum,
        NTAIL=ntail,
        BLOCK_M=block_m,
        BLOCK_P=block_p,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )
    if write_out is not out:
        out.copy_(write_out)
    return out


def _compute_linear_combination_out(input, coefficients, *, out=None):
    logger.debug("GEMS _COMPUTE_LINEAR_COMBINATION_OUT")
    return _compute_linear_combination(input, coefficients, out=out)
