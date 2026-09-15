import pytest
import torch

from .tensor_split import tensor_split


def assert_tensor_split_matches(input, indices_or_sections, dim=0):
    actual = tensor_split(input, indices_or_sections, dim)
    expected = torch.tensor_split(input, indices_or_sections, dim)

    assert len(actual) == len(expected)
    for actual_part, expected_part in zip(actual, expected):
        torch.testing.assert_close(actual_part, expected_part)
        assert actual_part.shape == expected_part.shape
        assert actual_part.is_contiguous()
        if actual_part.numel() != 0:
            assert actual_part.data_ptr() != input.data_ptr()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.int32]
)
@pytest.mark.parametrize(
    ("shape", "sections", "dim"),
    [
        ((5, 4), 3, 0),
        ((2, 1025), 2, 0),
        ((2, 3, 4), 5, -1),
    ],
)
def test_tensor_split_sections(dtype, shape, sections, dim):
    input = torch.arange(
        torch.tensor(shape).prod().item(), dtype=dtype, device="cpu"
    ).reshape(shape)

    assert_tensor_split_matches(input, sections, dim)


@pytest.mark.parametrize(
    ("indices", "dim"),
    [
        ([2, 5], 1),
        ((2, 5), 1),
        ([4, 2], 0),
        ([-1, 3], 0),
        ([2, 10], 0),
        ([], -1),
    ],
)
def test_tensor_split_indices(indices, dim):
    input = torch.arange(24, dtype=torch.float32, device="cpu").reshape(4, 6)

    assert_tensor_split_matches(input, indices, dim)


@pytest.mark.parametrize(
    ("indices_or_sections", "dim"),
    [
        (torch.tensor([2, 5], dtype=torch.long, device="cpu"), 1),
        (torch.tensor(3, dtype=torch.long, device="cpu"), 0),
    ],
)
def test_tensor_split_tensor_indices(indices_or_sections, dim):
    input = torch.arange(24, dtype=torch.float32, device="cpu").reshape(4, 6)

    assert_tensor_split_matches(input, indices_or_sections, dim)


def test_tensor_split_non_contiguous_input():
    input = torch.arange(24, dtype=torch.float32, device="cpu").reshape(4, 6).T

    assert not input.is_contiguous()
    assert_tensor_split_matches(input, [1, 3], dim=0)


@pytest.mark.parametrize("indices_or_sections", [3, [0, 0], torch.tensor(3)])
def test_tensor_split_empty_input(indices_or_sections):
    input = torch.empty((2, 0, 3), dtype=torch.float32, device="cpu")

    assert_tensor_split_matches(input, indices_or_sections, dim=1)


def test_tensor_split_returns_copies():
    input = torch.arange(24, dtype=torch.float32, device="cpu").reshape(4, 6)
    output = tensor_split(input, [2, 5], dim=1)

    output[0].fill_(-1)

    assert input.eq(torch.arange(24, dtype=torch.float32).reshape(4, 6)).all()


@pytest.mark.parametrize(
    ("indices_or_sections", "error_type"),
    [
        (True, TypeError),
        ([True, False], TypeError),
        ([1.5], TypeError),
        ((1.5,), TypeError),
        (torch.tensor(True), RuntimeError),
        (torch.tensor([True, False]), RuntimeError),
        (torch.tensor(1.5), RuntimeError),
        (torch.tensor([1.5]), RuntimeError),
        (torch.tensor(1, dtype=torch.int32), RuntimeError),
        (torch.tensor([1], dtype=torch.int32), RuntimeError),
        (torch.tensor([[1]], dtype=torch.long), RuntimeError),
    ],
)
def test_tensor_split_invalid_indices_or_sections(indices_or_sections, error_type):
    input = torch.arange(6, dtype=torch.float32, device="cpu")

    with pytest.raises(error_type):
        torch.tensor_split(input, indices_or_sections)
    with pytest.raises(error_type):
        tensor_split(input, indices_or_sections)


@pytest.mark.parametrize("indices_or_sections", [0, -1])
def test_tensor_split_invalid_sections(indices_or_sections):
    input = torch.arange(6, dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError):
        torch.tensor_split(input, indices_or_sections)
    with pytest.raises(RuntimeError):
        tensor_split(input, indices_or_sections)


@pytest.mark.parametrize("indices_or_sections", [1, [1], torch.tensor(1)])
def test_tensor_split_rejects_scalar_input(indices_or_sections):
    input = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError):
        torch.tensor_split(input, indices_or_sections)
    with pytest.raises(RuntimeError):
        tensor_split(input, indices_or_sections)


@pytest.mark.parametrize("dim", [-3, 2])
def test_tensor_split_invalid_dim(dim):
    input = torch.arange(6, dtype=torch.float32, device="cpu").reshape(2, 3)

    with pytest.raises(IndexError):
        torch.tensor_split(input, 2, dim)
    with pytest.raises(IndexError):
        tensor_split(input, 2, dim)
