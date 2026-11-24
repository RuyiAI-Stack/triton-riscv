import pytest
import torch

from .inverse import inverse


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "matrix, expected",
    [
        ([[0, 2], [1, 3]], [[-1.5, 1], [0.5, 0]]),
        (
            [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
            [[0.75, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.75]],
        ),
    ],
)
def test_inverse(matrix, expected, dtype):
    x = torch.tensor(matrix, dtype=dtype)
    ref = torch.tensor(expected, dtype=dtype)
    tri = inverse(x)
    tol = 1e-5 if dtype == torch.float32 else 1e-10

    torch.testing.assert_close(tri, ref, rtol=tol, atol=tol)
    identity = torch.eye(x.shape[-1], dtype=dtype)
    torch.testing.assert_close(x @ tri, identity, rtol=tol, atol=tol)
    torch.testing.assert_close(tri @ x, identity, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("transpose", [False, True])
def test_inverse_batched(dtype, transpose):
    matrix = torch.tensor([[0, 2], [1, 3]], dtype=dtype)
    expected = torch.tensor([[-1.5, 1], [0.5, 0]], dtype=dtype)
    x = torch.stack((matrix, 2 * matrix))
    ref = torch.stack((expected, expected / 2))
    if transpose:
        x = x.mT
        ref = ref.mT
    tri = inverse(x)
    tol = 1e-5 if dtype == torch.float32 else 1e-10

    torch.testing.assert_close(tri, ref, rtol=tol, atol=tol)
    identity = torch.eye(2, dtype=dtype).expand_as(x)
    torch.testing.assert_close(x @ tri, identity, rtol=tol, atol=tol)
    torch.testing.assert_close(tri @ x, identity, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "matrix, zero_pivot",
    [
        ([[0]], 1),
        ([[0, 0], [0, 0]], 1),
        ([[1, 2], [2, 4]], 2),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 0]], 3),
        ([[0, 1, 2], [0, 2, 4], [1, 0, 0]], 3),
    ],
)
def test_inverse_singular(matrix, zero_pivot, dtype):
    x = torch.tensor(matrix, dtype=dtype)

    with pytest.raises(
        torch.linalg.LinAlgError,
        match=rf"diagonal element {zero_pivot} is zero.*singular",
    ):
        inverse(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("singular_index", [0, 3, 5])
def test_inverse_singular_mixed_batch(dtype, transpose, singular_index):
    x = torch.eye(2, dtype=dtype).repeat(2, 3, 1, 1)
    x.reshape(-1, 2, 2)[singular_index] = torch.tensor([[1, 2], [2, 4]], dtype=dtype)
    if transpose:
        x = x.mT

    with pytest.raises(
        torch.linalg.LinAlgError,
        match=rf"\(Batch element {singular_index}\).*diagonal element 2.*singular",
    ):
        inverse(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_small_nonzero_pivot(dtype):
    # A small pivot is still invertible; singularity must not use a tolerance.
    x = torch.tensor([[2.0**-40, 0], [0, 1]], dtype=dtype)
    expected = torch.tensor([[2.0**40, 0], [0, 1]], dtype=dtype)

    torch.testing.assert_close(inverse(x), expected, rtol=0, atol=0)
