import torch
import triton
import triton.language as tl

# rot90: aten::rot90 as a Triton value copy. The upstream FlagGems kernel
# branched on a *runtime* ``k_norm`` scalar, which aborts codegen on this
# backend (rule: no runtime scalar if-chains in kernels); here ``K_NORM`` is
# a constexpr so each rotation gets its own specialization with the branch
# resolved at compile time.
#
# Deviations from upstream, each pinned to the aten oracle (probed on
# torch 2.15.0+cpu):
# - aten validates ``dims`` (size == 2, total dims >= 2, dim0 != dim1, dim
#   range) and normalizes negative dims; upstream did none of that and its
#   general (dims != [0, 1]) path was broken two ways: negative dims built a
#   permutation with out-of-range entries, and the permuted input was fed to
#   a flat-addressed kernel without materializing it first (reads storage
#   order, not logical order). This port adds the aten checks, normalizes
#   dims, and inserts the missing ``.contiguous()``/permuted output shape.
# - aten normalizes any integer k (k = 5 and k = -1 are legal, equal k % 4);
#   upstream's ((k % 4) + 4) % 4 normalization is kept as-is because it
#   already matches aten.

_BLOCK_SIZE = 1024


@triton.jit(do_not_specialize=["n_elements", "M", "N"])
def rot90_kernel_2d(
    in_ptr,
    out_ptr,
    n_elements,
    M,
    N,
    K_NORM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Rotate the [0, 1] plane of a contiguous [M, N, ...] tensor by K_NORM.

    Formulas (from upstream, verified against aten):
    - K_NORM == 0 (identity): out[i, j] = in[i, j]
    - K_NORM == 1 (90 degrees): out[i, j] = in[j, N - 1 - i]
    - K_NORM == 2 (180 degrees): out[i, j] = in[M - 1 - i, N - 1 - j]
    - K_NORM == 3 (270 degrees): out[i, j] = in[M - 1 - j, i]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m_minus_1 = M - 1
    n_minus_1 = N - 1
    # Product of the trailing dims: the rotation plane is [0, 1], the rest
    # of the layout is an inner run of ``inner`` contiguous elements.
    # (Upstream's formulas dropped this term, so its kernel was only correct
    # for exactly-2D inputs; aten requires the general rank here.)
    inner = n_elements // (M * N)

    if K_NORM == 0:
        # Identity case - output same shape as input [M, N, ...]
        out_dim0 = offsets // (N * inner)
        remainder = offsets % (N * inner)
        out_dim1 = remainder // inner

        in_dim0 = out_dim0
        in_dim1 = out_dim1

    elif K_NORM == 1:
        # 90 degrees - output shape [N, M, ...]
        out_dim0 = offsets // (M * inner)
        remainder = offsets % (M * inner)
        out_dim1 = remainder // inner

        # out[i, j] = in[j, N - 1 - i] where i = out_dim0, j = out_dim1
        in_dim0 = out_dim1
        in_dim1 = n_minus_1 - out_dim0

    elif K_NORM == 2:
        # 180 degrees - output same shape as input [M, N, ...]
        out_dim0 = offsets // (N * inner)
        remainder = offsets % (N * inner)
        out_dim1 = remainder // inner

        # out[i, j] = in[M - 1 - i, N - 1 - j]
        in_dim0 = m_minus_1 - out_dim0
        in_dim1 = n_minus_1 - out_dim1

    else:  # K_NORM == 3
        # 270 degrees - output shape [N, M, ...]
        out_dim0 = offsets // (M * inner)
        remainder = offsets % (M * inner)
        out_dim1 = remainder // inner

        # out[i, j] = in[M - 1 - j, i]
        in_dim0 = m_minus_1 - out_dim1
        in_dim1 = out_dim0

    tail = offsets % inner
    in_offset = in_dim0 * (N * inner) + in_dim1 * inner + tail

    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def rot90_2d(inp, k, dims, out):
    """Rotate the [0, 1] plane of a contiguous tensor into ``out``."""
    M = inp.shape[dims[0]]
    N = inp.shape[dims[1]]
    n_elements = out.numel()
    if n_elements == 0:
        return

    k_norm = ((k % 4) + 4) % 4

    grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
    rot90_kernel_2d[grid](
        inp,
        out,
        n_elements,
        M,
        N,
        K_NORM=k_norm,
        BLOCK_SIZE=_BLOCK_SIZE,
    )


def rot90(input, k=1, dims=[0, 1]):
    """Rotate an n-D tensor by 90 degrees ``k`` times in the plane ``dims``."""
    if len(dims) != 2:
        raise RuntimeError(
            f"expected total rotation dims == 2, but got dims = {len(dims)}"
        )
    ndim = input.dim()
    if ndim < 2:
        raise RuntimeError(f"expected total dims >= 2, but got total dims = {ndim}")
    if dims[0] == dims[1]:
        raise RuntimeError(
            f"expected rotation dims to be different, but got dim0 = {dims[0]} "
            f"and dim1 = {dims[1]}"
        )
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {d})"
            )

    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    dim0 = dims[0] % ndim
    dim1 = dims[1] % ndim

    # Normalize k to 0, 1, 2, 3 (aten does the same, any integer k is legal)
    k_norm = ((k % 4) + 4) % 4

    # For k=0 or k=2, output shape is same as input; for k=1 or k=3 the two
    # rotation dims are swapped.
    out_shape = list(x.shape)
    if k_norm == 1 or k_norm == 3:
        out_shape[dim0] = x.shape[dim1]
        out_shape[dim1] = x.shape[dim0]

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    if dim0 == 0 and dim1 == 1:
        # Direct path for dims = [0, 1]
        rot90_2d(x, k, dims, out)
    else:
        # General case: transpose to bring dims to [0, 1], rotate, transpose
        # back. The permuted input is materialized because the kernel
        # addresses it with flat contiguous indexing (upstream skipped this
        # and read storage order), and the transposed output is allocated
        # with the permuted shape (upstream reused the unpermuted shape).
        perm = [dim0, dim1]
        for i in range(ndim):
            if i != dim0 and i != dim1:
                perm.append(i)

        # Inverse permutation to restore original order
        inverse_perm = [0] * ndim
        inverse_perm[dim0] = 0
        inverse_perm[dim1] = 1
        idx = 2
        for i in range(ndim):
            if i != dim0 and i != dim1:
                inverse_perm[i] = idx
                idx += 1

        # Transpose, rotate 2D plane, transpose back
        x_transposed = x.permute(perm).contiguous()
        out_transposed = torch.empty(
            [out_shape[d] for d in perm], device=x.device, dtype=x.dtype
        )
        rot90_2d(x_transposed, k, [0, 1], out_transposed)
        out.copy_(out_transposed.permute(inverse_perm))

    return out
