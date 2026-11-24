from .cholesky_solve import cholesky_solve


def _cholesky_solve_helper(B, L, upper):
    return cholesky_solve(B, L, upper=upper)


__all__ = ["_cholesky_solve_helper"]
