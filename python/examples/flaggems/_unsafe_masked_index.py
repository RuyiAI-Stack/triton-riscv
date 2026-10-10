import torch
import triton
import triton.language as tl


@triton.jit
def _unsafe_masked_index_kernel(
    self_ptr,
    mask_ptr,
    offsets_ptr,
    fill_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < numel

    mask_vals = tl.load(mask_ptr + offsets, mask=active, other=0)
    gather_offsets = tl.load(offsets_ptr + offsets, mask=active, other=0).to(tl.int64)
    fill_val = tl.load(fill_ptr)
    gathered = tl.load(self_ptr + gather_offsets, mask=active & mask_vals, other=0)
    result = tl.where(mask_vals, gathered, fill_val)
    tl.store(out_ptr + offsets, result, mask=active)


def _unsafe_masked_index(self, mask, indices, fill):
    if not indices:
        raise ValueError("indices cannot be empty or None")
    if any(index is None for index in indices):
        raise TypeError("expected Tensor as element of indices")
    if len(indices) > self.ndim:
        raise IndexError("too many indices for tensor")

    broadcasted_indices = torch.broadcast_tensors(*indices)
    index_shape = tuple(broadcasted_indices[0].shape)
    trailing_shape = tuple(self.shape[len(indices) :])
    output_shape = index_shape + trailing_shape
    output = torch.empty(output_shape, dtype=self.dtype, device=self.device)
    if output.numel() == 0:
        return output

    linear_offsets = torch.zeros(index_shape, dtype=torch.int64, device=self.device)
    for dim, index in enumerate(broadcasted_indices):
        normalized = torch.where(index < 0, index + self.shape[dim], index)
        linear_offsets = linear_offsets + normalized * self.stride(dim)

    if trailing_shape:
        trailing_offsets = torch.zeros(
            trailing_shape, dtype=torch.int64, device=self.device
        )
        for trailing_index, size in enumerate(trailing_shape):
            view_shape = [1] * len(trailing_shape)
            view_shape[trailing_index] = size
            coords = torch.arange(size, device=self.device, dtype=torch.int64).reshape(
                view_shape
            )
            trailing_offsets = trailing_offsets + coords * self.stride(
                len(indices) + trailing_index
            )
        linear_offsets = linear_offsets.reshape(
            index_shape + (1,) * len(trailing_shape)
        )
        linear_offsets = linear_offsets + trailing_offsets.reshape(
            (1,) * len(index_shape) + trailing_shape
        )

    mask_broadcast = mask.expand(output_shape).contiguous().reshape(-1)
    offsets_flat = linear_offsets.contiguous().reshape(-1)
    fill_tensor = torch.as_tensor(fill, dtype=self.dtype, device=self.device).reshape(
        ()
    )

    block_size = 256
    grid = (triton.cdiv(output.numel(), block_size),)
    _unsafe_masked_index_kernel[grid](
        self,
        mask_broadcast,
        offsets_flat,
        fill_tensor,
        output,
        output.numel(),
        BLOCK_SIZE=block_size,
    )
    return output
