import torch
import triton
import triton.language as tl


@triton.jit
def _resize_output_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _resize_output(inp: torch.Tensor, size: list[int], device: torch.device):
    """Resize the input tensor to the specified size.

    Args:
        inp: Input tensor
        size: Target size as a list of integers
        device: Target device

    Returns:
        Resized tensor
    """

    # Calculate total number of elements
    old_numel = inp.numel()
    new_numel = 1
    for s in size:
        new_numel *= s

    # If same shape and device, just return input (no-op)
    if new_numel == old_numel and inp.device == device:
        return inp.reshape(size)

    # Create output tensor on target device
    out = torch.empty(size, dtype=inp.dtype, device=device)

    # Copy data - copy min(old_numel, new_numel) elements using Triton kernel
    copy_numel = min(old_numel, new_numel)
    if copy_numel > 0:
        # Flatten and slice to get the elements to copy
        inp_flat = inp.reshape(-1)[:copy_numel]
        out_flat = out.reshape(-1)[:copy_numel]
        # Use copy_ which uses Triton kernel
        from .copy import copy_

        copy_(out_flat, inp_flat)

    return out


def _resize_output_(inp: torch.Tensor, size: list[int], device: torch.device):
    """In-place resize the input tensor to the specified size.

    Args:
        inp: Input tensor (will be modified in place)
        size: Target size as a list of integers
        device: Target device

    Returns:
        Resized tensor (same object as input)
    """

    # Calculate total number of elements
    old_numel = inp.numel()
    new_numel = 1
    for s in size:
        new_numel *= s

    # For in-place, we use PyTorch's resize_ which handles all cases
    # This includes same device/different shape, different device, etc.
    if inp.device == device:
        # Same device - use PyTorch's resize_ which modifies in place
        inp.resize_(size)
        return inp
    else:
        # Different device - need to create temp and copy
        out = torch.empty(size, dtype=inp.dtype, device=device)
        copy_numel = min(old_numel, new_numel)
        if copy_numel > 0:
            inp_flat = inp.reshape(-1)[:copy_numel]
            out_flat = out.reshape(-1)[:copy_numel]
            from .copy import copy_

            copy_(out_flat, inp_flat)
        inp.resize_(size)
        inp.copy_(out)
        return inp
