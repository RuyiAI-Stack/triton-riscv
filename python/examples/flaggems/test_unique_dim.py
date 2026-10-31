import pytest
import torch

from .unique_dim import _build_composite_key, _remap_info, unique_dim


def test_unique_dim_values_inverse_counts():
    x = torch.tensor([[1, 2], [1, 2], [3, 4]], dtype=torch.int64, device="cpu")

    out = unique_dim(x, dim=0, sorted=True, return_inverse=True, return_counts=True)
    ref = torch.unique(x, dim=0, sorted=True, return_inverse=True, return_counts=True)

    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("zero_column", [0, 1])
def test_unique_dim_signed_zero(dtype, zero_column):
    x = torch.tensor([[-0.0, 1.0], [0.0, 1.0], [2.0, 1.0]], dtype=dtype)
    if zero_column == 1:
        x = x.flip(1)
    result = unique_dim(x, dim=0, sorted=True, return_inverse=True, return_counts=True)
    expected = torch.unique(
        x, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("first", [False, True])
def test_unique_dim_signed_zero_keys(dtype, first):
    # Equal zeros must share a sorting key, including the subsequent-column path.
    x = torch.tensor([[-0.0], [0.0], [-1.0], [1.0]], dtype=dtype)
    view, remap, offset = _remap_info(x)
    indices = None if first else torch.tensor([1, 0, 2, 3])
    groups = None if first else torch.zeros(4, dtype=torch.int64)
    keys = _build_composite_key(view, 0, indices, groups, 4, 1, offset, 1 << 32, remap)
    assert keys[0] == keys[1]
    assert keys[2] < keys[0] < keys[3]


@pytest.mark.parametrize("num_rows", [5, 9])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("dim", [0, 1])
def test_unique_dim_non_power_of_two_rows(num_rows, dtype, dim):
    # Repeated prefixes force the composite-key group scan between columns.
    rows = torch.tensor(
        [[2, 3], [0, 1], [2, 3], [-0.0, 1], [0, 2], [2, 4], [2, 3], [-1, 0], [0, 1]],
        dtype=dtype,
    )[:num_rows].clone()
    x = rows if dim == 0 else rows.t()
    actual = unique_dim(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    expected = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)
