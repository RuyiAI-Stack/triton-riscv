import torch
import triton
import triton.language as tl

_INVERSE_KERNEL_MAX_N = {torch.float32: 64, torch.float64: 32}


@triton.jit
def _inverse_kernel(
    A_ptr,
    Out_ptr,
    Info_ptr,
    n,
    BLOCK: tl.constexpr,
):
    """Invert one matrix per program via pivoted Gauss-Jordan elimination.

    The working matrix ``W`` and the inverse accumulator ``X`` are both held in
    registers and receive every row swap, so partial pivoting is tracked exactly
    (the augmented ``[A | I]`` scheme with both blocks kept on chip). Rows and
    columns are extracted with ``tl.gather`` (cheaper than block-wide
    where-reductions, especially in fp64), and column ``k`` is pinned to ``e_k``
    after each elimination step to stop rounding error from accumulating in the
    already-eliminated column.
    """
    pid = tl.program_id(0)
    base = pid * n * n
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    idx2d = rows[:, None] * n + cols[None, :]
    mask = (rows[:, None] < n) & (cols[None, :] < n)
    r2 = rows[:, None]
    c2 = cols[None, :]
    diag2d = r2 == c2

    # Row-index vectors for tl.gather: ``row_at`` selects one row of a block
    # (axis 0), ``col_at`` one column (axis 1).
    row_at = tl.full((1, BLOCK), 0, tl.int32)
    col_at = tl.full((BLOCK, 1), 0, tl.int32)

    # Non-power-of-two sizes are padded to BLOCK with the identity, making the
    # padded system block diagonal; the leading n x n block is then the inverse.
    W = tl.load(A_ptr + base + idx2d, mask=mask, other=0.0)
    W = tl.where(mask, W, tl.where(diag2d, 1.0, 0.0))
    X = tl.where(diag2d, 1.0, 0.0).to(A_ptr.dtype.element_ty)
    info = tl.full((), 0, tl.int32)

    for k in range(n):
        is_k = r2 == k
        is_c = c2 == k

        # Partial pivoting: p = argmax_{i >= k} |W[i, k]|.
        col_k = tl.sum(tl.gather(W, col_at + k, axis=1), axis=1)
        pick = tl.where(rows >= k, tl.abs(col_k), -1.0)
        p = tl.argmax(pick, axis=0)
        is_p = r2 == p

        # Swap rows k and p in both blocks (p == k is a no-op).
        row_k_w = tl.sum(tl.gather(W, row_at + k, axis=0), axis=0)
        row_p_w = tl.sum(tl.gather(W, row_at + p, axis=0), axis=0)
        row_k_x = tl.sum(tl.gather(X, row_at + k, axis=0), axis=0)
        row_p_x = tl.sum(tl.gather(X, row_at + p, axis=0), axis=0)
        W = tl.where(is_p, row_k_w[None, :], tl.where(is_k, row_p_w[None, :], W))
        X = tl.where(is_p, row_k_x[None, :], tl.where(is_k, row_p_x[None, :], X))

        # Normalize the pivot row by its diagonal entry.
        pivot = tl.sum(tl.where(is_k & is_c, W, 0.0))
        zero_pivot = pivot == 0
        # Keep the first failing pivot (1-based), as in ATen's linalg info.
        info = tl.where((info == 0) & zero_pivot, k + 1, info)
        # The wrapper rejects this matrix; avoid dividing by zero meanwhile.
        inv_pivot = 1.0 / tl.where(zero_pivot, 1.0, pivot)
        W = tl.where(is_k, W * inv_pivot, W)
        X = tl.where(is_k, X * inv_pivot, X)

        # Eliminate column k on every other row: row_i -= W[i, k] * row_k.
        factor = tl.sum(tl.gather(W, col_at + k, axis=1), axis=1)
        pivot_row_w = tl.sum(tl.gather(W, row_at + k, axis=0), axis=0)
        pivot_row_x = tl.sum(tl.gather(X, row_at + k, axis=0), axis=0)
        W = tl.where(is_k, W, W - factor[:, None] * pivot_row_w[None, :])
        X = tl.where(is_k, X, X - factor[:, None] * pivot_row_x[None, :])

        # Column k is exactly e_k now; pin it to keep the dead column exact.
        W = tl.where(is_c, tl.where(is_k, 1.0, 0.0), W)

    tl.store(Out_ptr + base + idx2d, X, mask=mask)
    tl.store(Info_ptr + pid, info)


def _inverse_kernel_impl(A):
    n = A.shape[-1]
    batch_shape = A.shape[:-2]
    batch = 1
    for dim in batch_shape:
        batch *= dim

    if batch == 0:
        return torch.empty_like(A)

    # Both buffers must be freshly materialized: `empty_like(A)` keeps the
    # input's strides, so for a non-contiguous operand the 2-D reshape below
    # would be a strided view while the kernel addresses flat `n*n` tiles.
    A_work = A.reshape(batch, n, n).contiguous()
    out_work = torch.empty((batch, n, n), dtype=A.dtype, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)

    block = max(triton.next_power_of_2(n), 16)
    # The per-step gathers and rank-1 update are latency bound; more warps
    # shorten the block-wide reductions. Measured optima: one warp up to
    # BLOCK 16, four for 32 (fp64 halves the per-lane register budget), and
    # eight for the 48/64-wide tiles.
    if block <= 16:
        num_warps = 1
    elif block <= 32:
        num_warps = 4
    else:
        num_warps = 8
    _inverse_kernel[(batch,)](
        A_work, out_work, info, n, BLOCK=block, num_warps=num_warps
    )
    torch.ops.aten._linalg_check_errors(info, "linalg.inv", is_matrix=A.dim() == 2)
    return out_work.reshape(A.shape)


def inverse(A):
    """Invert a square matrix (batched), equivalent to ``torch.inverse``.

    Args:
        A: tensor of shape (*, n, n) with floating point or complex dtype.

    Returns:
        A tensor of the same shape holding the inverse of every matrix.

    Raises:
        torch.linalg.LinAlgError: If any matrix in the input is singular.
    """

    assert A.dim() >= 2, "inverse: input must be at least 2D"
    assert A.shape[-1] == A.shape[-2], "inverse: input must be a square matrix"
    assert A.dtype in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ), f"inverse: unsupported dtype {A.dtype}"

    n = A.shape[-1]
    if not A.numel():
        return A.clone()
    use_kernel = (
        A.dtype in _INVERSE_KERNEL_MAX_N and n <= _INVERSE_KERNEL_MAX_N[A.dtype]
    )

    if use_kernel:
        return _inverse_kernel_impl(A)

    # Larger real matrices spill the register tile, and complex inputs are not
    # implemented in the kernel, so both are routed through the vendor
    # (cuSOLVER) implementation. The `.out` overload is used so the call does
    # not re-enter the FlagGems `aten::inverse` registration.
    out = torch.empty_like(A)
    return torch.ops.aten.inverse.out(A, out=out)
