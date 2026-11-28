import logging

import torch
import triton
import triton.language as tl

# x ** p is decomposed as exp2(p * log2(x)) (upstream note: tl pow is too
# heavy); both lower bit-exactly in fp64 on this toolchain (probed).
logger = logging.getLogger(__name__)


@triton.jit
def pairwise_distance_general_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, p, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, tl.exp2(p * tl.log2(diff)), 0.0), axis=1)
    dist = tl.exp2((1.0 / p) * tl.log2(acc))
    tl.store(out_ptr + pid, dist[:, None], row_mask)


@triton.jit
def pairwise_distance_p2_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, (diff * diff), 0.0), axis=1)

    dist = tl.sqrt(acc)
    tl.store(out_ptr + pid, dist[:, None], row_mask)


@triton.jit
def pairwise_distance_p1_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, diff, 0.0), axis=1)

    tl.store(out_ptr + pid, acc[:, None], row_mask)


@triton.jit
def pairwise_distance_p0_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    acc = tl.zeros([BLOCK_M], dtype=acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, (diff != 0).to(acc_dtype), 0.0), axis=1)

    tl.store(out_ptr + pid, acc[:, None], row_mask)


@triton.jit
def pairwise_distance_max_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = pid < N
    max_val = tl.full([BLOCK_M], -float("inf"), acc_dtype)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        max_val = tl.maximum(
            max_val, tl.max(tl.where(mask, diff, -float("inf")), axis=1)
        )
    tl.store(out_ptr + pid, max_val[:, None], row_mask)


@triton.jit
def pairwise_distance_min_kernel(
    x1_ptr, x2_ptr, out_ptr, N, D, eps, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    acc_dtype = tl.float64 if x1_ptr.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    min_val = tl.full([BLOCK_M], float("inf"), acc_dtype)
    row_mask = pid < N
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D)[None, :]
        col_mask = cols < D
        mask = row_mask & col_mask
        a = tl.load(x1_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        b = tl.load(x2_ptr + pid * D + cols, mask=mask, other=0).to(acc_dtype)
        diff = tl.abs(a - b + eps)
        min_val = tl.minimum(
            min_val, tl.min(tl.where(mask, diff, float("inf")), axis=1)
        )
    tl.store(out_ptr + pid, min_val[:, None], row_mask)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS PAIRWISE_DISTANCE")
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    D = x1.shape[-1]

    # Empty feature dim: torch returns 0 for finite p; inf/-inf have no
    # identity element over an empty reduction and torch raises (probed).
    # Short-circuit here to also keep the kernels away from D == 0 grids.
    if D == 0:
        if p == float("inf") or p == float("-inf"):
            raise RuntimeError(
                "pairwise_distance cannot compute the inf/-inf norm on an "
                "empty reduction dimension (no identity element)"
            )
        out = torch.zeros(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
        if keepdim:
            out = out.unsqueeze(-1)
        return out

    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)

    # Empty batch (probed): torch returns an empty output; nothing to launch.
    if N == 0:
        return out

    # Fixed launch heuristics (upstream tunes BLOCK_M/BLOCK_D per config;
    # this repo has no tuning runtime).
    BLOCK_M = 4
    BLOCK_D = min(triton.next_power_of_2(D), 1024)
    grid = (triton.cdiv(N, BLOCK_M),)

    if p == 2.0:
        pairwise_distance_p2_kernel[grid](x1, x2, out, N, D, eps, BLOCK_M, BLOCK_D)
    elif p == 1.0:
        pairwise_distance_p1_kernel[grid](x1, x2, out, N, D, eps, BLOCK_M, BLOCK_D)
    elif p == 0.0:
        pairwise_distance_p0_kernel[grid](x1, x2, out, N, D, eps, BLOCK_M, BLOCK_D)
    elif p == float("inf"):
        pairwise_distance_max_kernel[grid](x1, x2, out, N, D, eps, BLOCK_M, BLOCK_D)
    elif p == float("-inf"):
        pairwise_distance_min_kernel[grid](x1, x2, out, N, D, eps, BLOCK_M, BLOCK_D)
    else:
        pairwise_distance_general_kernel[grid](
            x1, x2, out, N, D, eps, p, BLOCK_M, BLOCK_D
        )

    return out
