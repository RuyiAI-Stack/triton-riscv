import torch
import triton
import triton.language as tl


@triton.jit
def nan_to_num__kernel(
    x_ptr,
    n_elements,
    nan_value,
    posinf_value,
    neginf_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    pos_mask = x == float("inf")
    neg_mask = x == float("-inf")
    nan_mask = x != x
    y = tl.where(
        nan_mask,
        nan_value,
        tl.where(pos_mask, posinf_value, tl.where(neg_mask, neginf_value, x)),
    ).to(x.dtype)

    tl.store(x_ptr + offsets, y, mask=mask)


def nan_to_num_(x, nan=0.0, posinf=None, neginf=None):
    # aten semantics: posinf/neginf default to the dtype's finite extremes.
    if (
        not x.dtype.is_floating_point
        or x.dtype == torch.float64
        or nan != 0.0
        or posinf is not None
        or neginf is not None
    ):
        # fp64 finfo extremes overflow the fp32 scalar kernel arguments,
        # custom replacements follow aten semantics exactly, and integer
        # inputs are an aten no-op; the aten in-place op keeps the
        # in-place identity contract.
        return torch.ops.aten.nan_to_num_(x, nan, posinf, neginf)

    pos = posinf if posinf is not None else torch.finfo(x.dtype).max
    neg = neginf if neginf is not None else torch.finfo(x.dtype).min
    n_elements = x.numel()
    if n_elements == 0:
        return x
    x_c = x if x.is_contiguous() else x.contiguous()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    nan_to_num__kernel[grid](x_c, n_elements, nan, pos, neg, BLOCK_SIZE=BLOCK_SIZE)
    if x_c is not x:
        # Non-contiguous inputs run on a packed copy; copy back to keep the
        # in-place contract (mutate and return the caller's tensor).
        x.copy_(x_c)
    return x
