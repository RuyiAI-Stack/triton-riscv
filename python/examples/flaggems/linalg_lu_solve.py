import math

import torch
import triton
import triton.language as tl

GATHER_MAX_N = 32
GATHER_MAX_RHS_TILE = 32


@triton.jit
def linalg_lu_solve_gather_kernel(
    LU,
    B,
    X,
    pivots,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_lu,
    stride_lu_row,
    stride_lu_col,
    batch_stride_piv,
    stride_piv_row,
    batch_stride_b,
    stride_b_row,
    stride_b_col,
    batch_stride_x,
    stride_x_row,
    stride_x_col,
    BLOCK_N: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Register-resident LU solve for one batch element and one RHS tile.

    The whole [N, BLOCK_RHS] system stays in registers for the entire solve,
    so each serial row step is a constant-index tl.gather (one warp shuffle)
    plus a rank-1 update instead of the per-row scalar reduction a textbook
    substitution loop performs. Both triangular phases are pre-scaled forms:

      forward  (unit-diagonal L): w_r -= L[r, i] * w_i        for r > i
      backward (U):               w_r -= U[r, i] * inv[r] * w_i for r < i

    with w initialized to inv_diag * y before the backward sweep, so the
    gathered w_i is already the final solution component at step i. This
    removes the O(N^2) scalar loads of the elementwise formulation: the factor
    is now read one full column per step, coalesced across the row axis.
    """
    batch_idx = tl.program_id(0)
    rhs_tile_idx = tl.program_id(1)

    rows = tl.arange(0, BLOCK_N)
    rows_mask = rows < N
    cols = rhs_tile_idx * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    col_mask = cols < nrhs

    lu_base = LU + batch_idx * batch_stride_lu
    piv_base = pivots + batch_idx * batch_stride_piv
    b_base = B + batch_idx * batch_stride_b
    x_base = X + batch_idx * batch_stride_x

    perm = rows
    for i in range(N):
        p = tl.load(piv_base + i * stride_piv_row).to(tl.int32) - 1
        at_i = rows == i
        at_p = rows == p
        v_i = tl.sum(tl.where(at_i, perm, 0))
        v_p = tl.sum(tl.where(at_p, perm, 0))
        perm = tl.where(at_i, v_p, tl.where(at_p, v_i, perm))

    w = tl.load(
        b_base + perm[:, None] * stride_b_row + cols[None, :] * stride_b_col,
        mask=rows_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    for i in range(N):
        l_col = tl.load(
            lu_base + rows * stride_lu_row + i * stride_lu_col,
            mask=(rows > i) & rows_mask,
            other=0.0,
        )
        w_i = tl.gather(w, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
        w = w - l_col[:, None] * w_i

    diag = tl.load(
        lu_base + rows * stride_lu_row + rows * stride_lu_col,
        mask=rows_mask,
        other=1.0,
    )
    inv_diag = 1.0 / diag
    inv_diag = inv_diag * (2.0 - diag * inv_diag)

    w = w * inv_diag[:, None]
    for i in range(N - 1, -1, -1):
        u_col = tl.load(
            lu_base + rows * stride_lu_row + i * stride_lu_col,
            mask=rows < i,
            other=0.0,
        )
        w_i = tl.gather(w, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
        w = w - (u_col * inv_diag)[:, None] * w_i

    tl.store(
        x_base + rows[:, None] * stride_x_row + cols[None, :] * stride_x_col,
        w,
        mask=rows_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def linalg_lu_solve_permute_kernel(
    B,
    X,
    pivots,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_piv,
    stride_piv_row,
    batch_stride_b,
    stride_b_row,
    stride_b_col,
    batch_stride_x,
    stride_x_row,
    stride_x_col,
    BLOCK_N: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
):
    """Write X = P @ B for the blocked path.

    PyTorch stores pivots as sequential 1-indexed row swaps, so composing them
    is serial in N. Only the row indices take part in that chain, so it runs on
    a register-resident index vector and the payload moves in one coalesced
    gathered pass -- far cheaper than N global row-exchange read/write pairs.
    """
    batch_idx = tl.program_id(0)
    rhs_tile_idx = tl.program_id(1)

    rows = tl.arange(0, BLOCK_N)
    rows_mask = rows < N
    cols = rhs_tile_idx * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    col_mask = cols < nrhs

    piv_base = pivots + batch_idx * batch_stride_piv

    perm = rows
    for i in range(N):
        p = tl.load(piv_base + i * stride_piv_row).to(tl.int32) - 1
        at_i = rows == i
        at_p = rows == p
        v_i = tl.sum(tl.where(at_i, perm, 0))
        v_p = tl.sum(tl.where(at_p, perm, 0))
        perm = tl.where(at_i, v_p, tl.where(at_p, v_i, perm))

    vals = tl.load(
        B
        + batch_idx * batch_stride_b
        + perm[:, None] * stride_b_row
        + cols[None, :] * stride_b_col,
        mask=rows_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.store(
        X
        + batch_idx * batch_stride_x
        + rows[:, None] * stride_x_row
        + cols[None, :] * stride_x_col,
        vals,
        mask=rows_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def linalg_lu_solve_blocked_kernel(
    LU,
    X,
    N: tl.constexpr,
    nrhs: tl.constexpr,
    batch_stride_lu,
    stride_lu_row,
    stride_lu_col,
    batch_stride_x,
    stride_x_row,
    stride_x_col,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    IS_DOUBLE: tl.constexpr,
):
    """Blocked in-place TRSM pair for large systems, operating on X = P @ B.

    Each program owns one batch element and one RHS tile, so programs never
    share an X column and only need intra-program ordering. Per diagonal block
    the serial gather chain runs over BLOCK_K rows only; the bulk of the update
    is hoisted into tl.dot panel products, which is what recovers throughput
    once N outgrows the register-resident gather kernel.

    Row blocks are visited on an aligned grid and masked, so N need not be a
    multiple of BLOCK_K: padded rows load as zero and contribute nothing.
    """
    batch_idx = tl.program_id(0)
    rhs_tile_idx = tl.program_id(1)

    lu_base = LU + batch_idx * batch_stride_lu
    x_base = X + batch_idx * batch_stride_x

    k_offsets = tl.arange(0, BLOCK_K)
    m_offsets = tl.arange(0, BLOCK_M)
    cols = rhs_tile_idx * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    col_mask = cols < nrhs
    num_blocks = tl.cdiv(N, BLOCK_K)

    for kb in range(num_blocks):
        k = kb * BLOCK_K
        rows_k = k + k_offsets
        valid_k = rows_k < N
        w = tl.load(
            x_base + rows_k[:, None] * stride_x_row + cols[None, :] * stride_x_col,
            mask=valid_k[:, None] & col_mask[None, :],
            other=0.0,
        )
        for i in range(BLOCK_K):
            l_col = tl.load(
                lu_base + rows_k * stride_lu_row + (k + i) * stride_lu_col,
                mask=(k_offsets > i) & valid_k,
                other=0.0,
            )
            w_i = tl.gather(w, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            w = w - l_col[:, None] * w_i

        tl.store(
            x_base + rows_k[:, None] * stride_x_row + cols[None, :] * stride_x_col,
            w,
            mask=valid_k[:, None] & col_mask[None, :],
        )

        for m in range(k + BLOCK_K, N, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < N
            tail = tl.load(
                x_base + rows_m[:, None] * stride_x_row + cols[None, :] * stride_x_col,
                mask=rows_m_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            if IS_DOUBLE:
                for kk in range(BLOCK_K):
                    l_col = tl.load(
                        lu_base + rows_m * stride_lu_row + (k + kk) * stride_lu_col,
                        mask=rows_m_mask & (k + kk < N),
                        other=0.0,
                    )
                    w_row = tl.gather(w, tl.full([1, BLOCK_RHS], kk, tl.int32), 0)
                    tail = tail - l_col[:, None] * w_row
            else:
                l_tile = tl.load(
                    lu_base
                    + rows_m[:, None] * stride_lu_row
                    + rows_k[None, :] * stride_lu_col,
                    mask=rows_m_mask[:, None] & valid_k[None, :],
                    other=0.0,
                )
                tail = tail - tl.dot(l_tile, w, input_precision="ieee")
            tl.store(
                x_base + rows_m[:, None] * stride_x_row + cols[None, :] * stride_x_col,
                tail,
                mask=rows_m_mask[:, None] & col_mask[None, :],
            )

    for kb in range(num_blocks - 1, -1, -1):
        k = kb * BLOCK_K
        rows_k = k + k_offsets
        valid_k = rows_k < N
        # CPU programs execute sequentially; preceding loads/stores are ordered.
        w = tl.load(
            x_base + rows_k[:, None] * stride_x_row + cols[None, :] * stride_x_col,
            mask=valid_k[:, None] & col_mask[None, :],
            other=0.0,
        )

        diag = tl.load(
            lu_base + rows_k * stride_lu_row + rows_k * stride_lu_col,
            mask=valid_k,
            other=1.0,
        )
        inv_diag = 1.0 / diag
        inv_diag = inv_diag * (2.0 - diag * inv_diag)

        w = w * inv_diag[:, None]
        for i in range(BLOCK_K - 1, -1, -1):
            u_col = tl.load(
                lu_base + rows_k * stride_lu_row + (k + i) * stride_lu_col,
                mask=(k_offsets < i) & valid_k & (k + i < N),
                other=0.0,
            )
            w_i = tl.gather(w, tl.full([1, BLOCK_RHS], i, tl.int32), 0)
            w = w - (u_col * inv_diag)[:, None] * w_i

        tl.store(
            x_base + rows_k[:, None] * stride_x_row + cols[None, :] * stride_x_col,
            w,
            mask=valid_k[:, None] & col_mask[None, :],
        )

        for m in range(0, k, BLOCK_M):
            rows_m = m + m_offsets
            rows_m_mask = rows_m < k
            head = tl.load(
                x_base + rows_m[:, None] * stride_x_row + cols[None, :] * stride_x_col,
                mask=rows_m_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            if IS_DOUBLE:
                for kk in range(BLOCK_K):
                    u_col = tl.load(
                        lu_base + rows_m * stride_lu_row + (k + kk) * stride_lu_col,
                        mask=rows_m_mask & (k + kk < N),
                        other=0.0,
                    )
                    w_row = tl.gather(w, tl.full([1, BLOCK_RHS], kk, tl.int32), 0)
                    head = head - u_col[:, None] * w_row
            else:
                u_tile = tl.load(
                    lu_base
                    + rows_m[:, None] * stride_lu_row
                    + rows_k[None, :] * stride_lu_col,
                    mask=rows_m_mask[:, None] & valid_k[None, :],
                    other=0.0,
                )
                head = head - tl.dot(u_tile, w, input_precision="ieee")
            tl.store(
                x_base + rows_m[:, None] * stride_x_row + cols[None, :] * stride_x_col,
                head,
                mask=rows_m_mask[:, None] & col_mask[None, :],
            )


def _can_use_gather_path(N, nrhs):
    """Use the register-resident gather kernel for matrices up to GATHER_MAX_N."""
    return N <= GATHER_MAX_N


def _get_gather_block_rhs(nrhs):
    return max(min(triton.next_power_of_2(nrhs), GATHER_MAX_RHS_TILE), 1)


def _get_gather_launch_config(dtype, N):
    if N >= 16:
        return {"num_warps": 4, "num_stages": 2}
    if dtype == torch.float64:
        return {"num_warps": 2, "num_stages": 1}
    return {"num_warps": 1, "num_stages": 2}


def _get_blocked_config(dtype, N):
    """Choose diagonal and panel tile sizes for the blocked solve kernel."""
    if dtype == torch.float64:
        return {
            "BLOCK_K": 16,
            "BLOCK_M": 32,
            "num_warps": 4,
            "num_stages": 1,
        }
    return {
        "BLOCK_K": 32,
        "BLOCK_M": 128 if N >= 128 else 64,
        "num_warps": 4,
        "num_stages": 1,
    }


def _get_blocked_block_rhs(nrhs):
    return 16


def linalg_lu_solve(LU, pivots, B, *, left=True, adjoint=False):
    """
    Solve the linear system AX = B using the LU factorization of A.

    Args:
        LU: (*, n, n) compact LU factorization matrix
        pivots: (*, n) pivot indices (1-indexed)
        B: (*, n, k) right-hand side matrix
        left: must be True; right-sided solves are not implemented
        adjoint: must be False; adjoint solves are not implemented

    Returns:
        X: (*, n, k) solution matrix

    LU, pivots, and B must have matching batch shapes; broadcasting is not
    implemented.
    """

    if not left or adjoint:
        raise NotImplementedError(
            "linalg_lu_solve only supports left=True and adjoint=False"
        )
    if LU.ndim < 2 or B.ndim < 2:
        raise ValueError("LU and B must be at least 2D")
    if LU.shape[-1] != LU.shape[-2]:
        raise ValueError("LU must be a square matrix")
    if B.shape[:-2] != LU.shape[:-2] or pivots.shape != LU.shape[:-1]:
        raise NotImplementedError(
            "linalg_lu_solve requires matching batch shapes for LU, pivots, and B"
        )
    n = LU.shape[-1]
    if B.shape[-2] != n:
        raise ValueError("B must have compatible dimensions with LU")
    if LU.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"linalg_lu_solve only supports float32/float64, got {LU.dtype}"
        )

    if LU.numel() == 0 or B.numel() == 0:
        return B.clone()

    batch = math.prod(LU.shape[:-2])
    nrhs = B.shape[-1]

    LU_work = LU.reshape(batch, n, n).contiguous()
    piv_work = pivots.reshape(batch, n).contiguous()

    with torch.no_grad():
        if _can_use_gather_path(n, nrhs):
            block_n = triton.next_power_of_2(n)
            block_rhs = _get_gather_block_rhs(nrhs)
            launch = _get_gather_launch_config(LU.dtype, n)
            B_work = B.reshape(batch, n, nrhs).contiguous()
            X = torch.empty_like(B_work)
            grid = (batch, triton.cdiv(nrhs, block_rhs))
            linalg_lu_solve_gather_kernel[grid](
                LU_work,
                B_work,
                X,
                piv_work,
                n,
                nrhs,
                LU_work.stride(0),
                LU_work.stride(1),
                LU_work.stride(2),
                piv_work.stride(0),
                piv_work.stride(1),
                B_work.stride(0),
                B_work.stride(1),
                B_work.stride(2),
                X.stride(0),
                X.stride(1),
                X.stride(2),
                BLOCK_N=block_n,
                BLOCK_RHS=block_rhs,
                **launch,
            )
        else:
            cfg = _get_blocked_config(LU.dtype, n)
            block_rhs = _get_blocked_block_rhs(nrhs)
            block_n = triton.next_power_of_2(n)
            B_work = B.reshape(batch, n, nrhs).contiguous()
            X = torch.empty_like(B_work)
            permute_rhs = _get_gather_block_rhs(nrhs)
            linalg_lu_solve_permute_kernel[(batch, triton.cdiv(nrhs, permute_rhs))](
                B_work,
                X,
                piv_work,
                n,
                nrhs,
                piv_work.stride(0),
                piv_work.stride(1),
                B_work.stride(0),
                B_work.stride(1),
                B_work.stride(2),
                X.stride(0),
                X.stride(1),
                X.stride(2),
                BLOCK_N=block_n,
                BLOCK_RHS=permute_rhs,
                num_warps=4,
                num_stages=1,
            )
            linalg_lu_solve_blocked_kernel[(batch, triton.cdiv(nrhs, block_rhs))](
                LU_work,
                X,
                n,
                nrhs,
                LU_work.stride(0),
                LU_work.stride(1),
                LU_work.stride(2),
                X.stride(0),
                X.stride(1),
                X.stride(2),
                BLOCK_K=cfg["BLOCK_K"],
                BLOCK_M=cfg["BLOCK_M"],
                BLOCK_RHS=block_rhs,
                IS_DOUBLE=(LU.dtype == torch.float64),
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )
    return X.reshape(B.shape)
