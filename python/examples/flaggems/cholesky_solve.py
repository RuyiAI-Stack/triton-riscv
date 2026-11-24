from .linalg_solve_triangular import solve_triangular


def cholesky_solve(B, L, upper=False):
    # Solve L L^T X = B (lower) or U^T U X = B (upper).
    if upper:
        y = solve_triangular(L, B, upper=True, transpose=True)
        return solve_triangular(L, y, upper=True, transpose=False)
    y = solve_triangular(L, B, upper=False, transpose=False)
    return solve_triangular(L, y, upper=False, transpose=True)


__all__ = ["cholesky_solve"]
