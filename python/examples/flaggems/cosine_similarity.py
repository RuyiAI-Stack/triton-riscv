import torch
import triton
import triton.language as tl

_BLOCK_M = 8
_BLOCK_D = 256


@triton.jit
def cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    D,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One block of rows per program; reduce along the contiguous last dim D.
    # Accumulate in float32 (or float64 for double inputs) so half/bfloat16
    # dot products and norms do not lose precision during the reduction.
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid < N
    dot = tl.zeros([BLOCK_M], dtype=acc_dtype)
    norm1 = tl.zeros([BLOCK_M], dtype=acc_dtype)
    norm2 = tl.zeros([BLOCK_M], dtype=acc_dtype)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0.0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0.0).to(acc_dtype)
        dot += tl.sum(tl.where(mask, a * b, 0.0), axis=1)
        norm1 += tl.sum(tl.where(mask, a * a, 0.0), axis=1)
        norm2 += tl.sum(tl.where(mask, b * b, 0.0), axis=1)
    # torch clamps each L2 norm independently by eps before dividing:
    # cos = dot / (max(||x1||, eps) * max(||x2||, eps)).
    denom1 = tl.maximum(tl.sqrt(norm1), eps)
    denom2 = tl.maximum(tl.sqrt(norm2), eps)
    out = dot / (denom1 * denom2)
    # Clamp the result into [-1, 1]: only rounding overshoot past +-1 can
    # reach here.  The where-form keeps NaN inputs NaN (fmax-style
    # minimum/maximum would silently swallow them).
    out = tl.where(out > 1.0, 1.0, tl.where(out < -1.0, -1.0, out))
    tl.store(out_ptr + pid, out[:, None].to(out_ptr.type.element_ty), row_mask)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    assert x1.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"cosine_similarity: unsupported dtype {x1.dtype}"
    # Broadcast to a common shape, matching torch.cosine_similarity semantics.
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    ndim = x1.ndim
    dim = dim % ndim if ndim > 0 else 0

    # Move the reduction dim to the last axis so it is contiguous, then
    # flatten the remaining (batch) dims into a single row index.
    if dim != ndim - 1:
        x1 = torch.movedim(x1, dim, -1)
        x2 = torch.movedim(x2, dim, -1)
    x1 = x1.contiguous()
    x2 = x2.contiguous()

    D = x1.shape[-1]
    out_shape = x1.shape[:-1]
    N = x1.numel() // D if D > 0 else x1.numel()

    x1 = x1.reshape(N, D)
    x2 = x2.reshape(N, D)
    out = torch.empty((N,), dtype=x1.dtype, device=x1.device)

    if N == 0:
        return out.reshape(out_shape)

    grid = (triton.cdiv(N, _BLOCK_M),)
    cosine_similarity_kernel[grid](
        x1, x2, out, N, D, eps, BLOCK_M=_BLOCK_M, BLOCK_D=_BLOCK_D
    )

    return out.reshape(out_shape)
