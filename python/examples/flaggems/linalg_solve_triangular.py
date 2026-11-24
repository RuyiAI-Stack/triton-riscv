import triton
import triton.language as tl

from .linalg_lu_factor import _batch_count, _block_for, _check_batch_square


@triton.jit
def _trsm_kernel(
    A_ptr,
    X_ptr,
    n,
    nrhs,
    UPPER: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK: tl.constexpr,
    NRHS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, NRHS)
    mask_a = (rows[:, None] < n) & (rows[None, :] < n)
    mask_x = (rows[:, None] < n) & (cols[None, :] < nrhs)
    base_a = pid * n * n
    base_x = pid * n * nrhs
    A = tl.load(
        A_ptr + base_a + rows[:, None] * n + rows[None, :], mask=mask_a, other=0.0
    )
    X = tl.load(
        X_ptr + base_x + rows[:, None] * nrhs + cols[None, :], mask=mask_x, other=0.0
    )
    A = A.to(tl.float32)
    X = X.to(tl.float32)
    forward = (UPPER and TRANS) or ((not UPPER) and (not TRANS))
    for step in range(BLOCK):
        i = step if forward else (BLOCK - 1 - step)
        active = i < n
        if TRANS:
            coeff = tl.sum(tl.where(rows[None, :] == i, A, 0.0), axis=1)
        else:
            coeff = tl.sum(tl.where(rows[:, None] == i, A, 0.0), axis=0)
        diag = 1.0 if UNIT else tl.sum(tl.where(rows == i, coeff, 0.0))
        safe = tl.where(tl.abs(diag) > 1.0e-20, diag, 1.0)
        if forward:
            take = rows < i
        else:
            take = rows > i
        dots = tl.sum(tl.where(take[:, None], coeff[:, None] * X, 0.0), axis=0)
        xi = tl.sum(tl.where(rows[:, None] == i, X, 0.0), axis=0)
        xi = tl.where(active, (xi - dots) / safe, xi)
        X = tl.where((rows[:, None] == i) & active, xi[None, :], X)
    tl.store(
        X_ptr + base_x + rows[:, None] * nrhs + cols[None, :],
        X,
        mask=mask_x,
    )


def _prepare_rhs(A, B):
    n = A.shape[-1]
    vector = B.ndim == A.ndim - 1 and B.shape[-1] == n
    if vector:
        nrhs = 1
        batch_shape = A.shape[:-2]
        batch = _batch_count(batch_shape)
        rhs = B.reshape(batch, n, 1).contiguous()
    elif B.ndim >= 2 and B.shape[-2] == n:
        nrhs = B.shape[-1]
        batch_shape = A.shape[:-2]
        batch = _batch_count(batch_shape)
        rhs = B.reshape(batch, n, nrhs).contiguous()
        vector = False
    else:
        raise RuntimeError("triangular solve: B is incompatible with A")
    return batch, n, nrhs, rhs, vector


def solve_triangular(
    A, B, *, upper=False, transpose=False, unitriangular=False, left=True
):
    if not left:
        # A is on the right: X A = B is equivalent to A^T X^T = B^T.
        xt = solve_triangular(
            A.transpose(-1, -2),
            B.transpose(-1, -2),
            upper=not upper,
            transpose=False,
            unitriangular=unitriangular,
            left=True,
        )
        return xt.transpose(-1, -2).contiguous()
    n = _check_batch_square(A, "solve_triangular")
    batch, n, nrhs, rhs, vector = _prepare_rhs(A, B)
    work_a = A.reshape(batch, n, n).contiguous()
    X = rhs.clone()
    if batch > 0 and n > 0:
        _trsm_kernel[(batch,)](
            work_a,
            X,
            n,
            nrhs,
            UPPER=upper,
            TRANS=transpose,
            UNIT=unitriangular,
            BLOCK=_block_for(n),
            NRHS=_block_for(nrhs),
        )
    if vector:
        return X.reshape(B.shape)
    return X.reshape(B.shape)


def linalg_solve_triangular(A, B, *, upper=False, left=True, unitriangular=False):
    return solve_triangular(
        A, B, upper=upper, transpose=False, unitriangular=unitriangular, left=left
    )
