"""Masked accumulation for fully indexed tensors, using CPU Triton atomics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_accumulate(
    OUT,
    MASK,
    INDICES,
    VALUES,
    N,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < N
    enabled = tl.load(MASK + pos, valid, other=0).to(tl.int1) & valid
    offset = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE)):
        idx = tl.load(INDICES + d * N + pos, valid, other=0).to(tl.int64)
        idx = tl.where(idx < 0, idx + SHAPE[d], idx)
        enabled = enabled & (idx >= 0) & (idx < SHAPE[d])
        offset += idx * STRIDES[d]
    val = tl.load(VALUES + pos, valid, other=0)
    tl.atomic_add(OUT + offset, val, enabled)


def _unsafe_masked_index_put_accumulate(inp, mask, indices, values):
    if isinstance(indices, torch.Tensor):
        indices = [indices]
    if len(indices) != inp.ndim or any(i is None for i in indices):
        raise NotImplementedError(
            "This example requires one integer index tensor per input dimension"
        )
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    if any(i.dtype not in (torch.int32, torch.int64) for i in indices):
        raise TypeError("indices must be int32 or int64")
    shape = torch.broadcast_shapes(
        mask.shape, values.shape, *(i.shape for i in indices)
    )
    idx = torch.stack(
        [i.expand(shape).to(torch.int64).contiguous().reshape(-1) for i in indices]
    )
    enabled = mask.expand(shape).contiguous()
    dtype = torch.float32 if inp.dtype in (torch.float16, torch.bfloat16) else inp.dtype
    vals = values.expand(shape).to(dtype).contiguous()
    out = inp.to(dtype).clone(memory_format=torch.contiguous_format)
    n = enabled.numel()
    if n:
        _masked_accumulate[(triton.cdiv(n, 256),)](
            out, enabled, idx, vals, n, tuple(out.shape), tuple(out.stride()), BLOCK=256
        )
    return out.to(inp.dtype)
