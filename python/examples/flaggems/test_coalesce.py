import pytest
import torch

from .coalesce import coalesce


def _assert_coalesced(actual, expected):
    assert actual.layout == torch.sparse_coo
    assert actual.is_coalesced()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual.indices(), expected.indices(), rtol=0, atol=0)
    torch.testing.assert_close(actual.values(), expected.values(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ],
)
def test_coalesce(dtype):
    indices = torch.tensor([[2, 0, 2, 1, 0], [4, 1, 4, 0, 1]])
    values = torch.tensor([1, 0, 3, 4, 0], dtype=dtype)
    x = torch.sparse_coo_tensor(indices, values, (3, 5))

    _assert_coalesced(coalesce(x), x.coalesce())
    assert not x.is_coalesced()


@pytest.mark.parametrize(
    "shape",
    [(7,), (2, 3, 4), (2, 2, 3, 2), (2, 2, 2, 2, 3), (2, 2, 2, 2, 2, 2)],
)
def test_coalesce_sparse_dimensions(shape):
    indices = torch.tensor([[size - 1, 0, size - 1, 1, 0] for size in shape])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    x = torch.sparse_coo_tensor(indices, values, shape)

    _assert_coalesced(coalesce(x), x.coalesce())


def test_coalesce_noncontiguous():
    indices = torch.tensor([[2, 0, 2, 1, 0], [4, 1, 4, 0, 1]]).t().contiguous().t()
    values = torch.arange(10, dtype=torch.float64)[::2]
    x = torch.sparse_coo_tensor(indices, values, (3, 5))
    assert not x._indices().is_contiguous()
    assert not x._values().is_contiguous()

    _assert_coalesced(coalesce(x), x.coalesce())


def test_coalesce_multiple_superblocks():
    # Each scan superblock covers 2048 words of 32 keys; this needs two offsets.
    indices = torch.tensor([[65536, 31, 0, 65535, 65536, 32, 31]])
    values = torch.arange(1, 8, dtype=torch.float32)
    x = torch.sparse_coo_tensor(indices, values, (65537,))

    _assert_coalesced(coalesce(x), x.coalesce())


def test_coalesce_separate_superblock_scan():
    # 1025 superblocks require the separate scan, including its final offset.
    last_key = 1 << 26
    indices = torch.tensor([[last_key, 0, last_key]])
    values = torch.tensor([1.0, 2.0, 3.0])
    x = torch.sparse_coo_tensor(indices, values, (last_key + 1,))

    _assert_coalesced(coalesce(x), x.coalesce())


def test_coalesce_coalesced():
    x = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([2.0, 3.0]),
        (2, 2),
        is_coalesced=True,
    )

    assert coalesce(x) is x


def test_coalesce_empty():
    x = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.int64), torch.empty(0), (0, 3)
    )

    _assert_coalesced(coalesce(x), x.coalesce())


def test_coalesce_sparse_scalar():
    x = torch.sparse_coo_tensor(
        torch.empty((0, 3), dtype=torch.int64), torch.tensor([1.0, 2.0, 3.0]), ()
    )

    _assert_coalesced(coalesce(x), x.coalesce())
