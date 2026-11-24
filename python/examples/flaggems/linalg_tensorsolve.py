from .linalg_solve import linalg_solve


def linalg_tensorsolve(A, B, dims=None):
    if dims is not None:
        order = list(dims) + [d for d in range(A.ndim) if d not in dims]
        A = A.permute(order)
    left = B.numel()
    mat = A.reshape(left, A.numel() // left)
    return linalg_solve(mat, B.reshape(left)).reshape(A.shape[B.ndim :])


__all__ = ["linalg_tensorsolve"]
