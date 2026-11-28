import math

import torch
import triton
import triton.language as tl

# flipud: aten::flipud as a value copy. Flips along dim 0 (requires >= 1-d).
# Contiguous inputs mirror the row index; strided inputs decode through
# constexpr shape/stride tuples flipping dim 0. Complex inputs recurse
# through view_as_real (the flip of a real-pair view is the real-pair view
# of the flip). An autograd node keeps gradients flowing (flip is its own
# inverse).


@triton.jit
def flipud_contiguous_kernel(
    input,
    output,
    n_elements,
    height,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    row = offsets // row_size
    within = offsets % row_size
    # Rebuilt from the (row, within) decode rather than upstream's
    # `offsets + (height-1-2*row) * row_size`: this backend drops the
    # `offsets +` term when lowering that form (probed).
    src_row = height - 1 - row
    input_offsets = src_row * row_size + within
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def flipud_strided_kernel(
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
        input_index = SHAPE[dim] - 1 - index if dim == 0 else index
        input_offsets += input_index * INPUT_STRIDES[dim]
        output_offsets += index * OUTPUT_STRIDES[dim]

    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + output_offsets, values, mask=mask)


def _flipud_impl(self: torch.Tensor) -> torch.Tensor:
    if self.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")
    if self.is_complex():
        return torch.view_as_complex(_flipud_impl(torch.view_as_real(self)))
    output = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    if self.is_contiguous() and output.is_contiguous():
        row_size = math.prod(self.shape[1:])
        flipud_contiguous_kernel[grid](
            self,
            output,
            n_elements,
            self.shape[0],
            row_size,
            BLOCK_SIZE=1024,
        )
    else:
        flipud_strided_kernel[grid](
            self,
            output,
            n_elements,
            SHAPE=tuple(self.shape),
            INPUT_STRIDES=tuple(self.stride()),
            OUTPUT_STRIDES=tuple(output.stride()),
            NDIM=self.ndim,
            BLOCK_SIZE=1024,
        )
    return output


class Flipud(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return _flipud_impl(input)

    @staticmethod
    def backward(ctx, grad_output):
        return _flipud_impl(grad_output)


def flipud(self: torch.Tensor) -> torch.Tensor:
    """Value copy of ``self`` flipped along dim 0 (input must be >= 1-d)."""
    if torch.is_grad_enabled() and self.requires_grad:
        return Flipud.apply(self)
    return _flipud_impl(self)
