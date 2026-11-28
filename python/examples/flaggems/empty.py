import torch
import triton
import triton.language as tl

# empty: aten::empty -- uninitialized allocation. The allocation goes
# through empty_strided with strides computed on the meta device for the
# requested memory_format (upstream order, avoiding recursion through
# aten::empty). The fill kernel is part of the upstream file but is never
# launched: the storage is deliberately NOT initialized, matching aten.


@triton.jit
def empty_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, tl.zeros((BLOCK_SIZE,), tl.float32), mask=mask)


def empty(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """Uninitialized tensor of ``size``; never zero-filled, like aten::empty."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    if memory_format is None:
        memory_format = torch.contiguous_format
    shape = tuple(size)
    meta = torch.empty(shape, dtype=dtype, device="meta", memory_format=memory_format)
    return torch.empty_strided(
        shape,
        meta.stride(),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
