import torch
import triton
import triton.language as tl


@triton.jit
def _bound_kernel(
    values_ptr,
    bounds_ptr,
    out_ptr,
    n,
    nb,
    RIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    value = tl.load(values_ptr + offs, mask=mask, other=0).to(tl.float32)
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.full([BLOCK], nb, dtype=tl.int32)
    for _step in range(16):
        active = lo < hi
        mid = (lo + hi) // 2
        bound = tl.load(bounds_ptr + mid, mask=mid < nb, other=0).to(tl.float32)
        go_hi = bound <= value if RIGHT else bound < value
        lo = tl.where(active & go_hi, mid + 1, lo)
        hi = tl.where(active & ~go_hi, mid, hi)
    tl.store(out_ptr + offs, lo.to(tl.int64), mask=mask)


def _bounded(values, boundaries, right=False):
    flat = values.reshape(-1).contiguous()
    bounds = boundaries.contiguous().to(torch.float32)
    out = torch.empty(flat.shape, dtype=torch.int64, device=values.device)
    if flat.numel():
        block = 128
        _bound_kernel[(triton.cdiv(flat.numel(), block),)](
            flat.to(torch.float32),
            bounds,
            out,
            flat.numel(),
            bounds.numel(),
            RIGHT=right,
            BLOCK=block,
        )
    return out.view(values.shape)


def bucketize(input, boundaries, *, out_int32=False, right=False):
    result = _bounded(input, boundaries, right=right)
    return result.to(torch.int32) if out_int32 else result
