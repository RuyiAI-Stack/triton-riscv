import torch
import triton
import triton.language as tl


@triton.jit
def _unsafe_masked_index_kernel(
    self_ptr,
    mask_ptr,
    indices_ptr,
    out_ptr,
    N,
    fill_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lane = offsets < N

    # The lane mask only guards the tile edge: the gathered address is
    # dereferenced at masked-off lanes too (the "unsafe" contract -- mask
    # picks the result, it does not guard the load), so callers must keep
    # every index in [0, self.numel()).
    mask_val = tl.load(mask_ptr + offsets, mask=lane, other=0)
    indices_val = tl.load(indices_ptr + offsets, mask=lane, other=0)
    self_val = tl.load(self_ptr + indices_val, mask=lane, other=0.0)

    result = tl.where(mask_val, self_val, fill_val)
    tl.store(out_ptr + offsets, result, mask=lane)


def _unsafe_masked_index(self, mask, indices, fill):
    # out[j] = mask.flat[j] ? self.flat[indices[0].flat[j]] : fill, with the
    # output shaped like mask and typed like self (the inductor flat-lane
    # contract of aten::_unsafe_masked_index; equal to the eager aten
    # composite on 1-D inputs, probed on this host).
    if not indices or indices[0] is None:
        raise ValueError("indices cannot be empty or None")

    indices_tensor = indices[0]
    output_shape = mask.shape

    # contiguous by shape (not empty_like(mask)): empty_like preserves the
    # strides of a non-contiguous mask, and out.reshape(-1) would then be a
    # copy that silently swallows every kernel write
    out = torch.empty(output_shape, dtype=self.dtype, device=self.device)

    self_flat = self.reshape(-1)
    indices_flat = indices_tensor.reshape(-1)
    mask_flat = mask.reshape(-1)
    out_flat = out.reshape(-1)

    N = mask_flat.numel()

    if isinstance(fill, torch.Tensor):
        fill = fill.item() if fill.numel() == 1 else 0.0
    # keep the scalar exact for integer out dtypes (a float fill truncates
    # at the store anyway, like aten's masked_fill)
    fill_val = float(fill) if self.dtype.is_floating_point else int(fill)

    _unsafe_masked_index_kernel[(triton.cdiv(N, 512),)](
        self_flat,
        mask_flat,
        indices_flat,
        out_flat,
        N,
        fill_val,
        BLOCK_SIZE=512,
    )

    return out.reshape(output_shape)
