import torch
import triton
import triton.language as tl

# split_copy: aten::split as value copies. One Triton launch per chunk: a
# linear (period, chunk_numel) div/mod copy for contiguous inputs, a padded
# 8-slot strided gather otherwise. Rank capped at 8; validation messages
# mirror aten.

_BLOCK_SIZE = 1024
_MAX_RANK = 8


@triton.jit
def _split_copy_contig_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    src_begin,
    src_period,
    chunk_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    src_offsets = (
        (offsets // chunk_numel) * src_period + src_begin + offsets % chunk_numel
    )
    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _split_copy_strided_kernel(
    src_ptr,
    dst_ptr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
    n_elements,
    src_base,
    RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offsets = offsets.to(tl.int64)
    src_strides = [s0, s1, s2, s3, s4, s5, s6, s7]
    chunk_dims = [d0, d1, d2, d3, d4, d5, d6, d7]
    rem = offsets
    src_offsets = tl.full([], src_base, tl.int64)
    for j in tl.static_range(RANK - 1, -1, -1):
        idx = rem % chunk_dims[j]
        rem = rem // chunk_dims[j]
        src_offsets += idx * src_strides[j]
    values = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, values, mask=mask)


def split_copy(self: torch.Tensor, split_size: int, dim: int = 0):
    """Split into chunks of ``split_size`` copies along ``dim`` (last chunk may be short)."""
    if self.dim() == 0:
        raise RuntimeError("split expects at least a 1-dimensional tensor")
    if split_size < 0:
        raise RuntimeError(
            f"split expects split_size be non-negative, but got split_size={split_size}"
        )

    dim_size = self.size(dim)
    if split_size == 0 and dim_size != 0:
        raise RuntimeError(
            "split_size can only be 0 if dimension size is 0, but got dimension "
            f"size of {dim_size}"
        )
    dim = dim % self.dim()

    dim_stride = self.stride(dim)
    out_shape = list(self.shape)
    out_shape[dim] = min(split_size, dim_size)

    src_strides = [int(s) for s in self.stride()]
    dim_stride = src_strides[dim]
    period = dim_size * dim_stride
    # An empty dimension still splits into a single empty chunk (torch keeps
    # one output whose ``dim`` extent is 0). The upstream chunk-count
    # arithmetic would divide by zero for dim_size == 0 here; guarded to
    # match its stated intent and aten.
    n_chunks = 1 if dim_size == 0 else max(1, (dim_size + split_size - 1) // split_size)

    result = []
    chunk_start = 0
    for _ in range(n_chunks):
        chunk_len = min(split_size, dim_size - chunk_start)
        out_shape[dim] = chunk_len
        out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        n_elements = out.numel()

        if n_elements > 0:
            if self.is_contiguous():
                grid = (triton.cdiv(n_elements, 1024),)
                _split_copy_contig_kernel[grid](
                    self,
                    out,
                    n_elements,
                    chunk_start * dim_stride,
                    period,
                    chunk_len * dim_stride,
                    BLOCK_SIZE=_BLOCK_SIZE,
                )
            else:
                rank = self.dim()
                if rank > _MAX_RANK:
                    raise RuntimeError(
                        f"split_copy supports at most {_MAX_RANK} dimensions, "
                        f"got {rank}"
                    )
                chunk_strides = list(src_strides)
                chunk_dims = list(self.shape)
                chunk_dims[dim] = chunk_len
                chunk_strides = chunk_strides + [0] * (_MAX_RANK - len(chunk_strides))
                chunk_dims = chunk_dims + [1] * (_MAX_RANK - len(chunk_dims))

                src_base = chunk_start * dim_stride
                grid = (triton.cdiv(n_elements, 1024),)
                _split_copy_strided_kernel[grid](
                    self,
                    out,
                    *chunk_strides,
                    *chunk_dims,
                    n_elements,
                    src_base,
                    RANK=rank,
                    BLOCK_SIZE=_BLOCK_SIZE,
                )
        result.append(out)
        chunk_start += chunk_len

    return result
