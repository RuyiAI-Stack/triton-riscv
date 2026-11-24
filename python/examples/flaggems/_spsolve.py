import torch
import triton
import triton.language as tl

from .linalg_solve import linalg_solve


@triton.jit
def _csr_dense_kernel(crow, col, val, out, n_rows, n_cols):
    row = tl.program_id(0)
    start = tl.load(crow + row)
    end = tl.load(crow + row + 1)
    for idx in range(start, end):
        c = tl.load(col + idx)
        v = tl.load(val + idx)
        tl.store(out + row * n_cols + c, v)


def _spsolve(A, B):
    if not A.is_sparse_csr:
        A = A.to_sparse_csr()
    n = A.shape[0]
    dense = torch.zeros((n, n), dtype=torch.float32, device=A.device)
    _csr_dense_kernel[(n,)](A.crow_indices(), A.col_indices(), A.values(), dense, n, n)
    rhs = B if B.dtype == torch.float32 else B.to(torch.float32)
    return linalg_solve(dense, rhs)


__all__ = ["_spsolve"]
