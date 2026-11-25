"""Shared nearest / nearest-exact resize kernels for CPU targets."""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _resize(
    X,
    Y,
    N,
    INS: tl.constexpr,
    OUTS: tl.constexpr,
    SCALES: tl.constexpr,
    EXACT: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    volume: tl.constexpr = math.prod(OUTS)
    index = (pos // volume) * math.prod(INS)
    for axis in tl.static_range(len(INS)):
        output_step = math.prod(OUTS[axis + 1 :])
        input_step = math.prod(INS[axis + 1 :])
        coord = (pos // output_step) % OUTS[axis]
        source = tl.floor(
            (coord.to(tl.float32) + (0.5 if EXACT else 0.0)) * SCALES[axis]
        ).to(tl.int64)
        source = tl.minimum(source, INS[axis] - 1)
        index += source * input_step
    if BACKWARD:
        value = tl.load(X + pos, valid, other=0)
        tl.atomic_add(Y + index, value, valid)
    else:
        value = tl.load(X + index, valid, other=0)
        tl.store(Y + pos, value, valid)


def nearest(input, output_size, scales, exact=True, backward_size=None):
    d = input.ndim - 2
    if d not in (1, 2, 3):
        raise ValueError("expected one to three spatial dimensions")
    if output_size is None:
        if scales is None or any(s is None for s in scales):
            raise ValueError("size or scales required")
        output_size = tuple(math.floor(a * b) for a, b in zip(input.shape[2:], scales))
    output_size = tuple(output_size)
    if len(output_size) != d or any(s <= 0 for s in output_size):
        raise ValueError("invalid output size")
    ins = (
        tuple(backward_size[2:])
        if backward_size is not None
        else tuple(input.shape[2:])
    )
    scales = scales or (None,) * d
    reciprocal = tuple(
        1.0 / s if s is not None and s > 0 else a / b
        for a, b, s in zip(ins, output_size, scales)
    )
    if backward_size is None:
        result = torch.empty(
            (*input.shape[:2], *output_size), dtype=input.dtype, device=input.device
        )
    else:
        if tuple(input.shape) != (*backward_size[:2], *output_size):
            raise ValueError("gradient shape mismatch")
        acc = (
            torch.float32
            if input.dtype in (torch.float16, torch.bfloat16)
            else input.dtype
        )
        result = torch.zeros(tuple(backward_size), dtype=acc, device=input.device)
    n = input.numel() if backward_size is not None else result.numel()
    if n:
        _resize[(triton.cdiv(n, 256),)](
            input.contiguous(),
            result,
            n,
            ins,
            output_size,
            reciprocal,
            exact,
            backward_size is not None,
            BLOCK=256,
        )
    return result.to(input.dtype)
