from ._cpu_port_utils import pow_cpu

# Ported from FlagOpen/FlagGems 599c5ba05c86, src/flag_gems/ops/dist.py.

import logging
import torch
import triton
import triton.language as tl

exp2 = tl.exp2
log2 = tl.log2
lib_pow = pow_cpu
logger = logging.getLogger(__name__)
BLOCK_SIZE = 4096
SINGLE_KERNEL_THRESHOLD = 16384


@triton.jit
def _load_abs_diff(X, Y, idx, mask):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    a = tl.load(X + idx, mask=mask, other=0.0).to(acc_dtype)
    b = tl.load(Y + idx, mask=mask, other=0.0).to(acc_dtype)
    return tl.abs(a - b)


@triton.jit
def _pow_1_over_p(total, p):
    return lib_pow(total, 1.0 / p)


@triton.jit
def dist_p2_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff * diff, 0.0)
    tl.store(Out, tl.sqrt(tl.sum(acc)))


@triton.jit
def dist_p1_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, diff, 0.0)
    tl.store(Out, tl.sum(acc))


@triton.jit
def dist_p0_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, (diff != 0).to(acc_dtype), 0.0)
    tl.store(Out, tl.sum(acc))


@triton.jit
def dist_general_kernel(X, Y, Out, N, p, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc += tl.where(mask, exp2(p * log2(diff)), 0.0)
    tl.store(Out, _pow_1_over_p(tl.sum(acc), p))


@triton.jit
def dist_max_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.maximum(acc, tl.where(mask, diff, -float("inf")))
    tl.store(Out, tl.max(acc))


@triton.jit
def dist_min_kernel(X, Y, Out, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), float("inf"), dtype=acc_dtype)
    for start in range(0, N, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < N
        diff = _load_abs_diff(X, Y, idx, mask)
        acc = tl.minimum(acc, tl.where(mask, diff, float("inf")))
    tl.store(Out, tl.min(acc))


@triton.jit
def dist_p2_kernel_1(X, Y, Mid, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    diff = _load_abs_diff(X, Y, idx, mask)
    tl.store(Mid + pid, tl.sum(tl.where(mask, diff * diff, 0.0)))


@triton.jit
def dist_p2_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        acc += tl.where(mask, tl.load(Mid + idx, mask=mask, other=0.0), 0.0)
    tl.store(Out, tl.sqrt(tl.sum(acc)))


@triton.jit
def dist_p1_kernel_1(X, Y, Mid, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    diff = _load_abs_diff(X, Y, idx, mask)
    tl.store(Mid + pid, tl.sum(tl.where(mask, diff, 0.0)))


@triton.jit
def dist_p0_kernel_1(X, Y, Mid, N, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    diff = _load_abs_diff(X, Y, idx, mask)
    tl.store(Mid + pid, tl.sum(tl.where(mask, (diff != 0).to(acc_dtype), 0.0)))


@triton.jit
def dist_sum_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        acc += tl.where(mask, tl.load(Mid + idx, mask=mask, other=0.0), 0.0)
    tl.store(Out, tl.sum(acc))


@triton.jit
def dist_general_kernel_1(X, Y, Mid, N, p, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if X.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    diff = _load_abs_diff(X, Y, idx, mask)
    tl.store(Mid + pid, tl.sum(tl.where(mask, exp2(p * log2(diff)), 0.0)))


@triton.jit
def dist_general_kernel_2(Mid, Out, p, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    p = p.to(acc_dtype)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        acc += tl.where(mask, tl.load(Mid + idx, mask=mask, other=0.0), 0.0)
    tl.store(Out, _pow_1_over_p(tl.sum(acc), p))


@triton.jit
def dist_max_kernel_1(X, Y, Mid, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    diff = _load_abs_diff(X, Y, idx, mask)
    tl.store(Mid + pid, tl.max(tl.where(mask, diff, -float("inf"))))


@triton.jit
def dist_max_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), -float("inf"), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        m = tl.load(Mid + idx, mask=mask, other=0.0)
        acc = tl.maximum(acc, tl.where(mask, m, -float("inf")))
    tl.store(Out, tl.max(acc))


@triton.jit
def dist_min_kernel_1(X, Y, Mid, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    diff = _load_abs_diff(X, Y, idx, mask)
    tl.store(Mid + pid, tl.min(tl.where(mask, diff, float("inf"))))


@triton.jit
def dist_min_kernel_2(Mid, Out, MID_SIZE, BLOCK_SIZE: tl.constexpr):
    acc_dtype = tl.float64 if Mid.type.element_ty == tl.float64 else tl.float32
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((BLOCK_SIZE,), float("inf"), dtype=acc_dtype)
    for start in range(0, MID_SIZE, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < MID_SIZE
        m = tl.load(Mid + idx, mask=mask, other=0.0)
        acc = tl.minimum(acc, tl.where(mask, m, float("inf")))
    tl.store(Out, tl.min(acc))


def dist(input, other, p=2):
    logger.debug("GEMS DIST")
    if input.shape != other.shape:
        input, other = torch.broadcast_tensors(input, other)
    if not input.is_contiguous():
        input = input.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()
    n = input.numel()
    if n == 0:
        if p == float("inf") or p == float("-inf") or p < 0:
            raise RuntimeError(
                f"dist cannot compute the {p} norm on an empty tensor (no identity element over an empty reduction)"
            )
        return torch.zeros([], dtype=input.dtype, device=input.device)
    out = torch.empty([], dtype=input.dtype, device=input.device)
    p = float(p)
    if n <= SINGLE_KERNEL_THRESHOLD:
        block = triton.next_power_of_2(min(n, BLOCK_SIZE))
        if p == 2:
            dist_p2_kernel[1,](input, other, out, n, BLOCK_SIZE=block)
        elif p == 1:
            dist_p1_kernel[1,](input, other, out, n, BLOCK_SIZE=block)
        elif p == 0:
            dist_p0_kernel[1,](input, other, out, n, BLOCK_SIZE=block)
        elif p == float("inf"):
            dist_max_kernel[1,](input, other, out, n, BLOCK_SIZE=block)
        elif p == float("-inf"):
            dist_min_kernel[1,](input, other, out, n, BLOCK_SIZE=block)
        else:
            dist_general_kernel[1,](input, other, out, n, p, BLOCK_SIZE=block)
        return out
    mid_dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    mid_size = triton.cdiv(n, BLOCK_SIZE)
    mid = torch.empty(mid_size, dtype=mid_dtype, device=input.device)
    grid_1 = (mid_size,)
    grid_2 = (1,)
    if p == 2:
        dist_p2_kernel_1[grid_1](
            input, other, mid, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        dist_p2_kernel_2[grid_2](mid, out, mid_size, BLOCK_SIZE=BLOCK_SIZE)
    elif p == 1:
        dist_p1_kernel_1[grid_1](
            input, other, mid, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        dist_sum_kernel_2[grid_2](mid, out, mid_size, BLOCK_SIZE=BLOCK_SIZE)
    elif p == 0:
        dist_p0_kernel_1[grid_1](
            input, other, mid, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        dist_sum_kernel_2[grid_2](mid, out, mid_size, BLOCK_SIZE=BLOCK_SIZE)
    elif p == float("inf"):
        dist_max_kernel_1[grid_1](
            input, other, mid, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        dist_max_kernel_2[grid_2](mid, out, mid_size, BLOCK_SIZE=BLOCK_SIZE)
    elif p == float("-inf"):
        dist_min_kernel_1[grid_1](
            input, other, mid, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        dist_min_kernel_2[grid_2](mid, out, mid_size, BLOCK_SIZE=BLOCK_SIZE)
    else:
        dist_general_kernel_1[grid_1](
            input, other, mid, n, p, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        dist_general_kernel_2[grid_2](mid, out, p, mid_size, BLOCK_SIZE=BLOCK_SIZE)
    return out
