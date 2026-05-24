import torch
import triton
import triton.language as tl


@triton.jit
def flip_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def flip(A, dims):
    strides = list(A.stride())
    flip_dims_b = [False for _ in A.stride()]
    for dim in dims:
        assert dim >= -A.dim() and dim < A.dim(), (
            f"Dimension out of range (expected to be in range of [{-A.dim()}, {A.dim() - 1}], but got {dim})"
        )
        assert not flip_dims_b[dim], (
            f"dim {dim} appears multiple times in the list of dims"
        )
        flip_dims_b[dim] = True
    n = 0
    offset = 0
    for i in range(len(flip_dims_b)):
        if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
            offset += strides[i] * (A.shape[i] - 1)
            strides[i] = -strides[i]
            n += 1
    if n == 0 or A.numel() <= 1:
        return A.clone()

    # Use torch.flip to materialize the flipped view, then copy via kernel
    flipped = torch.flip(A, dims)
    out = torch.empty_like(A)
    total = A.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    flip_kernel[grid](
        flipped.contiguous(),
        out,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
