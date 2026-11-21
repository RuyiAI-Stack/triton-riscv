import torch
import triton
import triton.language as tl

# empty_permuted: aten::empty_permuted -- uninitialized allocation whose
# physical memory order follows ``physical_layout`` (the logical dims
# ordered outermost-to-innermost physically). The touch kernel loads and
# stores back the storage (no meaningful value is produced; the memory
# stays uninitialized, matching aten).


@triton.jit
def empty_permuted_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(output_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def _strides_from_physical_layout(size, physical_layout):
    # Walking physical_layout backwards and accumulating the sizes yields
    # the stride of each logical dimension.
    strides = [0] * len(size)
    accumulated = 1
    for dim in reversed(physical_layout):
        strides[dim] = accumulated
        accumulated *= size[dim]
    return strides


def _volume(shape):
    out = 1
    for s in shape:
        out *= s
    return out


def empty_permuted(
    size,
    physical_layout,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Uninitialized tensor of ``size`` laid out physically per ``physical_layout``."""
    shape = tuple(size)
    physical_layout = list(physical_layout)
    if len(physical_layout) != len(shape):
        raise RuntimeError(
            f"empty_permuted: physical_layout must have the same length as size, "
            f"but got {len(physical_layout)} and {len(shape)}"
        )
    if sorted(physical_layout) != list(range(len(shape))):
        raise RuntimeError(
            f"empty_permuted: physical_layout must be a permutation of "
            f"[0, {len(shape)}), but got {physical_layout}"
        )
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    out = torch.empty_strided(
        shape,
        _strides_from_physical_layout(shape, physical_layout),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    N = _volume(shape)
    if N > 0:
        grid = (triton.cdiv(N, 1024),)
        empty_permuted_kernel[grid](out, N, BLOCK_SIZE=1024)
    return out
