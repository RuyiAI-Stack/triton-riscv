import math

import torch
import triton
import triton.language as tl

# fliplr: aten::fliplr as a value copy. Flips along dim -2 (requires >= 2-d).
# Contiguous inputs take a fast kernel that mirrors the column index; strided
# inputs go through a constexpr shape/stride decode that flips dim 1.


@triton.jit
def fliplr_contiguous_kernel(
    input,
    output,
    n_elements,
    width,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    row = offsets // (width * inner_size)
    col = (offsets // inner_size) % width
    within = offsets % inner_size
    # Upstream mirrors via `offsets + (width-1-2*col) * inner_size`; this
    # backend drops the `offsets +` term when lowering that form (probed:
    # every output lane reads from width-1-2*k), so the source offset is
    # rebuilt from the full (row, col, within) decode instead.
    src_col = width - 1 - col
    input_offsets = row * width * inner_size + src_col * inner_size + within
    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def fliplr_strided_kernel(
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
        input_index = SHAPE[dim] - 1 - index if dim == 1 else index
        input_offsets += input_index * INPUT_STRIDES[dim]
        output_offsets += index * OUTPUT_STRIDES[dim]

    values = tl.load(input + input_offsets, mask=mask)
    tl.store(output + output_offsets, values, mask=mask)


def fliplr(self: torch.Tensor) -> torch.Tensor:
    """Value copy of ``self`` flipped along dim -2 (input must be >= 2-d)."""
    if self.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    output = torch.empty_like(self)
    n_elements = self.numel()
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, 1024),)
    if self.is_contiguous() and output.is_contiguous():
        inner_size = math.prod(self.shape[2:])
        fliplr_contiguous_kernel[grid](
            self,
            output,
            n_elements,
            self.shape[1],
            inner_size,
            BLOCK_SIZE=1024,
        )
    else:
        fliplr_strided_kernel[grid](
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
