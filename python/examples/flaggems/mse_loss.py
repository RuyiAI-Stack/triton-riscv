import math
from enum import Enum

import torch
import triton
import triton.language as tl


@triton.jit
def kernel_1(inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    target_ptrs = target + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(tl.float32)
    target_val = tl.load(target_ptrs, mask=mask, other=0).to(tl.float32)
    sub = inp_val - target_val
    pow_val = sub * sub
    if reduction == 1:
        sum_val = tl.sum(pow_val) / M
    else:
        sum_val = tl.sum(pow_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


@triton.jit
def mse_loss_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    res = (x - y) * (x - y)
    tl.store(out_ptr + offsets, res, mask=mask)


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def mse_loss(inp, target, reduction=Reduction.MEAN.value):
    if isinstance(reduction, str):
        if reduction == "none":
            reduction = 0
        elif reduction == "mean":
            reduction = 1
        elif reduction == "sum":
            reduction = 2

    if reduction == Reduction.NONE.value:
        inp_c = inp.contiguous()
        target_c = target.contiguous()
        out = torch.empty_like(inp_c)
        n_elements = inp_c.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        mse_loss_kernel[grid](inp_c, target_c, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    kernel_1[(mid_size, 1, 1)](inp, target, mid, M, block_size, reduction)
    kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out
