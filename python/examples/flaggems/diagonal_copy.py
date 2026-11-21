import torch
import triton
import triton.language as tl

# diagonal_copy: torch.diagonal as a value copy (not a view).
# torch.diagonal provides the strided view (host side); a Triton kernel
# walks the view via its shape/strides and copies the values into a fresh
# contiguous tensor. This replaces the upstream pointwise_dynamic
# instantiation machinery, which is not available here.


@triton.jit
def _diagonal_copy_kernel(
    input,
    output,
    n_elements,
    SHAPE: tl.constexpr,
    INPUT_STRIDES: tl.constexpr,
    OUTPUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets.to(tl.int64)
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    output_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    for dim in tl.static_range(NDIM - 1, -1, -1):
        index = linear % SHAPE[dim]
        linear //= SHAPE[dim]
        input_offsets += index * INPUT_STRIDES[dim]
        output_offsets += index * OUTPUT_STRIDES[dim]
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + output_offsets, values, mask=mask)


def diagonal_copy(
    self: torch.Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> torch.Tensor:
    """Diagonal of ``self`` as a fresh tensor (value copy, never a view)."""
    ndim = self.ndim
    dim1 = dim1 if dim1 >= 0 else dim1 + ndim
    dim2 = dim2 if dim2 >= 0 else dim2 + ndim
    if dim1 == dim2:
        raise ValueError("dim1 and dim2 must be different")

    diag_view = torch.diagonal(self, offset=offset, dim1=dim1, dim2=dim2)
    output = torch.empty(diag_view.shape, dtype=self.dtype, device=self.device)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    _diagonal_copy_kernel[grid](
        diag_view,
        output,
        n_elements,
        SHAPE=tuple(diag_view.shape),
        INPUT_STRIDES=tuple(diag_view.stride()),
        OUTPUT_STRIDES=tuple(output.stride()),
        NDIM=diag_view.ndim,
        BLOCK_SIZE=1024,
    )
    return output
