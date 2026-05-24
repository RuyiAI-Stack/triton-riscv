import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["mini", "maxi"])
def clip_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    mini,
    maxi,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result = tl.minimum(maxi, tl.maximum(mini, x_f32))
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@triton.jit(do_not_specialize=["mini"])
def clip_min_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    mini,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result = tl.maximum(mini, x_f32)
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@triton.jit(do_not_specialize=["maxi"])
def clip_max_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    maxi,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    result = tl.minimum(maxi, x_f32)
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


def _clip_launch(A, kernel, *args):
    A_c = A.contiguous()
    out = torch.empty_like(A_c)
    n_elements = A_c.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    kernel[grid](A_c, out, n_elements, *args, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(A)


def clip(A, mini=None, maxi=None):
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if maxi is None:
        return _clip_launch(A, clip_min_kernel, float(mini))
    if mini is None:
        return _clip_launch(A, clip_max_kernel, float(maxi))
    return _clip_launch(A, clip_kernel, float(mini), float(maxi))


def clip_(A, mini=None, maxi=None):
    result = clip(A, mini=mini, maxi=maxi)
    A.copy_(result)
    return A
