import pytest
import torch

from .linalg_lu_solve import linalg_lu_solve


def _make_lu_inputs(n, nrhs, dtype):
    # Known factors avoid requiring LAPACK to prepare the solve or its reference.
    eye = torch.eye(n, dtype=dtype)
    lower = eye + torch.tril(torch.ones_like(eye), diagonal=-1) / 16
    upper = 2 * eye + torch.triu(torch.ones_like(eye), diagonal=1) / 8
    lu = torch.tril(lower, diagonal=-1) + upper
    pivots = torch.arange(1, n + 1, dtype=torch.int32)
    pivots[:2] = n
    permutation = list(range(n))
    # Undo the sequential row swaps: P @ A = L @ U.
    for row in range(n - 1, -1, -1):
        pivot = int(pivots[row]) - 1
        permutation[row], permutation[pivot] = permutation[pivot], permutation[row]
    a = (lower @ upper)[permutation]
    expected = torch.arange(n * nrhs, dtype=dtype).reshape(n, nrhs) % 7 / 4 - 1
    b = a @ expected
    return a, lu, pivots, b, expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n,nrhs", [(3, 1), (32, 33), (33, 17)])
def test_linalg_lu_solve(n, nrhs, dtype):
    a, lu, pivots, b, expected = _make_lu_inputs(n, nrhs, dtype)

    actual = linalg_lu_solve(lu, pivots, b)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(a @ actual, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_linalg_lu_solve_batched(dtype, noncontiguous):
    a, lu, pivots, b, expected = _make_lu_inputs(3, 5, dtype)
    scales = torch.tensor([[1, 2], [3, 4]], dtype=dtype)[..., None, None]
    lu = lu.repeat(2, 2, 1, 1)
    pivots = pivots.repeat(2, 2, 1)
    b = b * scales
    expected = expected * scales
    if noncontiguous:
        lu = lu.mT.contiguous().mT
        b = b.mT.contiguous().mT
        pivots = pivots.transpose(-1, -2).contiguous().transpose(-1, -2)

    actual = linalg_lu_solve(lu, pivots, b)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    torch.testing.assert_close(a @ actual, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_lu_solve_empty_batch(dtype):
    lu = torch.empty((0, 3, 3), dtype=dtype)
    pivots = torch.empty((0, 3), dtype=torch.int32)
    b = torch.empty((0, 3, 2), dtype=dtype)

    torch.testing.assert_close(linalg_lu_solve(lu, pivots, b), b)


@pytest.mark.parametrize("options", [{"left": False}, {"adjoint": True}])
def test_linalg_lu_solve_unsupported_modes(options):
    _, lu, pivots, b, _ = _make_lu_inputs(3, 1, torch.float32)

    with pytest.raises(NotImplementedError, match="left=True and adjoint=False"):
        linalg_lu_solve(lu, pivots, b, **options)


def test_linalg_lu_solve_mismatched_batch_shapes():
    _, lu, pivots, b, _ = _make_lu_inputs(3, 1, torch.float32)
    lu = lu.repeat(2, 1, 1, 1)
    pivots = pivots.repeat(2, 1, 1)
    b = b.repeat(1, 2, 1, 1)

    with pytest.raises(NotImplementedError, match="matching batch shapes"):
        linalg_lu_solve(lu, pivots, b)
