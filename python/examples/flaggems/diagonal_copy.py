import torch
import triton
import triton.language as tl


@triton.jit
def diagonal_copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    dim1_size,
    dim2_size,
    diag_len,
    offset,
    outer_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    offs = offs.to(tl.int64)
    diag_idx = offs % diag_len
    outer_idx = offs // diag_len
    base = outer_idx * outer_stride

    offset_vec = tl.full([BLOCK_SIZE], offset, tl.int64)
    row_idx = tl.where(offset_vec >= 0, diag_idx, diag_idx - offset_vec)
    col_idx = tl.where(offset_vec >= 0, diag_idx + offset_vec, diag_idx)
    in_offsets = base + row_idx * dim2_size + col_idx

    x = tl.load(in_ptr + in_offsets, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


def diagonal_copy(
    self: torch.Tensor, offset: int = 0, dim1: int = 0, dim2: int = 1
) -> torch.Tensor:
    """
    Performs the same operation as torch.diagonal, but returns a copy instead of a view.
    """
    # Validate dimensions
    ndim = self.ndim
    dim1 = dim1 if dim1 >= 0 else dim1 + ndim
    dim2 = dim2 if dim2 >= 0 else dim2 + ndim

    if dim1 == dim2:
        raise ValueError("dim1 and dim2 must be different")

    x = self if self.is_contiguous() else self.contiguous()
    perm = [d for d in range(x.ndim) if d not in (dim1, dim2)] + [dim1, dim2]
    x_perm = x.permute(perm).contiguous()

    dim1_size = x_perm.shape[-2]
    dim2_size = x_perm.shape[-1]
    if offset >= 0:
        diag_len = min(dim1_size, dim2_size - offset)
    else:
        diag_len = min(dim1_size + offset, dim2_size)

    if diag_len <= 0:
        out_shape = list(x_perm.shape[:-2]) + [0]
        return torch.empty(out_shape, device=self.device, dtype=self.dtype)

    out = torch.empty(
        list(x_perm.shape[:-2]) + [int(diag_len)],
        device=self.device,
        dtype=self.dtype,
    )
    if out.numel() == 0:
        return out

    def grid(meta):
        return (triton.cdiv(out.numel(), meta["BLOCK_SIZE"]),)

    BLOCK_SIZE = 1024

    diagonal_copy_kernel[grid](
        x_perm,
        out,
        out.numel(),
        dim1_size,
        dim2_size,
        int(diag_len),
        offset,
        dim1_size * dim2_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
