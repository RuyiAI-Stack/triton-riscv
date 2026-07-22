import torch
import triton
import triton.language as tl


def volume(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


@triton.jit
def empty_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Empty kernel that does nothing - just allocates uninitialized memory."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Empty means uninitialized, so we don't write anything
    # But we need at least one store to make the kernel valid
    # Store a dummy value that the user will ignore anyway
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def empty(*size, dtype=None, layout=None, device=None, pin_memory=None):
    """Returns a tensor filled with uninitialized data."""
    if dtype is None:
        dtype = torch.get_default_dtype()

    out = torch.zeros(size, device=device, dtype=dtype)
    N = volume(size)

    def grid_fn(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    empty_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out
