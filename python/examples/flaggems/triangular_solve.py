from .detach_copy import _copy_values
from .linalg_solve_triangular import solve_triangular


def triangular_solve(B, A, upper=False, transpose=False, unitriangular=False):
    solution = solve_triangular(
        A,
        B,
        upper=upper,
        transpose=transpose,
        unitriangular=unitriangular,
        left=True,
    )
    return solution, _copy_values(A)
