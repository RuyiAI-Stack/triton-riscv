import math

import torch
import triton
import triton.language as tl


@triton.jit
def range_func(y_ptr, start, step, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    range_val = offsets * step + start
    tl.store(y_ptr + offsets, range_val, mask=mask)


def _range_size(start, end, step):
    # torch.range (deprecated) includes the end point: the element count is
    # the number of values start + i*step (computed in double) that stay
    # within [start, end] for a positive step, mirrored for a negative one.
    # The floor estimate is within one of that count; the two bounded
    # adjustments below absorb the difference at knife-edge float bounds
    # (e.g. torch.range(0, 0.3, 0.1) has 3 elements, torch.range(0, 1, 0.1)
    # has 11).
    def within(i):
        value = start + i * step
        return value <= end if step > 0 else value >= end

    i = max(0, math.floor((end - start) / step))
    bumps = 0
    while bumps < 2 and within(i + 1):
        i += 1
        bumps += 1
    trims = 0
    while trims < 2 and i > 0 and not within(i):
        i -= 1
        trims += 1
    return i + 1


def range(
    start, end, step: float | int = 1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    if dtype == torch.bfloat16:
        # Mirror the explicit upstream/CUDA rejection.
        raise RuntimeError("torch.range does not support bfloat16 on CUDA")
    if step == 0:
        raise RuntimeError("step must be nonzero")
    if (step > 0 and end < start) or (step < 0 and end > start):
        raise RuntimeError(
            "upper bound and lower bound inconsistent with step sign"
        )

    if dtype is None:
        # torch.range defaults to the default float dtype even for integer
        # bounds, unlike torch.arange (which picks int64).
        dtype = torch.get_default_dtype()

    if dtype in (torch.int32, torch.int64):
        start = int(start)
        end = int(end)
        step = int(step)
        size = (end - start) // step + 1
    else:
        size = _range_size(float(start), float(end), float(step))

    size = int(size)

    # BLOCK_SIZE=128 balances occupancy and grid launch overhead for
    # typical 1-D ranges (repo arange.py precedent).
    BLOCK_SIZE = 128
    grid = triton.cdiv(size, BLOCK_SIZE)

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = "cpu"

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    if size > 0:
        range_func[grid,](result, start, step, size, BLOCK_SIZE)
    return result
