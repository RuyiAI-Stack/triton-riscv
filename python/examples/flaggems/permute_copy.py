import torch
import triton
import triton.language as tl

# permute_copy: aten::permute as a value copy. Specialized Triton kernels
# for ranks 1-3 (the permutation is inverted with tl.where over scalar
# args); ranks above 3 use the permute-view + contiguous aten fallback
# (upstream behavior). 0-d inputs return a plain clone.


@triton.jit
def _permute_copy_kernel_1d(
    src_ptr,
    dst_ptr,
    n_elements,
    src_stride0,
    dst_stride0,
    perm0,
    rank,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    o0 = offsets
    perm_inv0 = perm0
    src_offset = tl.where(perm_inv0 == 0, o0, 0) * src_stride0
    dst_offset = o0 * dst_stride0
    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


@triton.jit
def _permute_copy_kernel_2d(
    src_ptr,
    dst_ptr,
    n_elements,
    dst_shape1,
    src_stride0,
    src_stride1,
    dst_stride0,
    dst_stride1,
    perm0,
    perm1,
    rank,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    o1 = offsets % dst_shape1
    o0 = offsets // dst_shape1
    perm_inv0 = tl.where(perm1 == 0, 1, 0)
    perm_inv1 = tl.where(perm0 == 1, 0, 1)
    src_offset = (
        tl.where(perm_inv0 == 0, o0, o1) * src_stride0
        + tl.where(perm_inv1 == 0, o0, o1) * src_stride1
    )
    dst_offset = o0 * dst_stride0 + o1 * dst_stride1
    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


@triton.jit
def _permute_copy_kernel_3d(
    src_ptr,
    dst_ptr,
    n_elements,
    dst_shape2,
    dst_shape1,
    src_stride0,
    src_stride1,
    src_stride2,
    dst_stride0,
    dst_stride1,
    dst_stride2,
    perm0,
    perm1,
    perm2,
    rank,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    temp = offsets
    o2 = temp % dst_shape2
    temp = temp // dst_shape2
    o1 = temp % dst_shape1
    o0 = temp // dst_shape1

    perm_inv0 = tl.where(perm1 == 0, 1, tl.where(perm2 == 0, 2, 0))
    perm_inv1 = tl.where(perm0 == 1, 0, tl.where(perm2 == 1, 2, 1))
    perm_inv2 = tl.where(perm0 == 2, 0, tl.where(perm1 == 2, 1, 2))

    src_offset = (
        tl.where(perm_inv0 == 0, o0, tl.where(perm_inv0 == 1, o1, o2)) * src_stride0
        + tl.where(perm_inv1 == 0, o0, tl.where(perm_inv1 == 1, o1, o2)) * src_stride1
        + tl.where(perm_inv2 == 0, o0, tl.where(perm_inv2 == 1, o1, o2)) * src_stride2
    )

    dst_offset = o0 * dst_stride0 + o1 * dst_stride1 + o2 * dst_stride2

    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


def permute_copy(x: torch.Tensor, dims):
    """Copy of ``x.permute(dims)``; Triton paths for ranks 1-3, aten fallback above."""
    ndim = x.ndim
    if ndim == 0:
        return x.clone()

    dims = [d if d >= 0 else d + ndim for d in dims]

    out_shape = [x.shape[d] for d in dims]

    if x.numel() == 0:
        return torch.empty(out_shape, dtype=x.dtype, device=x.device)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()

    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)

    if ndim == 1:
        _permute_copy_kernel_1d[grid](
            src,
            out,
            n_elements,
            src.stride(0),
            out.stride(0),
            dims[0],
            ndim,
            BLOCK_SIZE=1024,
        )
    elif ndim == 2:
        _permute_copy_kernel_2d[grid](
            src,
            out,
            n_elements,
            out.shape[1],
            src.stride(0),
            src.stride(1),
            out.stride(0),
            out.stride(1),
            dims[0],
            dims[1],
            ndim,
            BLOCK_SIZE=1024,
        )
    elif ndim == 3:
        _permute_copy_kernel_3d[grid](
            src,
            out,
            n_elements,
            out.shape[2],
            out.shape[1],
            src.stride(0),
            src.stride(1),
            src.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dims[0],
            dims[1],
            dims[2],
            ndim,
            BLOCK_SIZE=1024,
        )
    else:
        return x.permute(dims).contiguous()
    return out
