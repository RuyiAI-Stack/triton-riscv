import pytest
import torch

from ._unsafe_masked_index import _unsafe_masked_index


def test_unsafe_masked_index():
    src = torch.arange(1023, dtype=torch.float32, device="cpu")
    mask = torch.arange(1023, device="cpu").remainder(3).eq(0)
    indices = torch.arange(1023, dtype=torch.int64, device="cpu").flip(0)

    out = _unsafe_masked_index(src, mask, [indices], fill=-1.0)
    ref = torch._unsafe_masked_index(src, mask, [indices], fill=-1.0)

    torch.testing.assert_close(out, ref)


def test_unsafe_masked_index_multidimensional_indices():
    src = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 3, 4)
    mask = torch.tensor([[True, False], [False, True]], device="cpu").unsqueeze(-1)
    indices0 = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64, device="cpu")
    indices1 = torch.tensor([[2, 1], [0, 2]], dtype=torch.int64, device="cpu")

    out = _unsafe_masked_index(src, mask, [indices0, indices1], fill=-1.0)
    ref = torch._unsafe_masked_index(src, mask, [indices0, indices1], fill=-1.0)

    torch.testing.assert_close(out, ref)


def test_unsafe_masked_index_rejects_none_index():
    src = torch.arange(8, dtype=torch.float32, device="cpu")
    mask = torch.ones(8, dtype=torch.bool, device="cpu")

    with pytest.raises(TypeError):
        torch._unsafe_masked_index(src, mask, [None], fill=-1.0)

    with pytest.raises(TypeError):
        _unsafe_masked_index(src, mask, [None], fill=-1.0)


def test_unsafe_masked_index_strided_source_and_broadcast_indices():
    src = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32, device="cpu").reshape(
        2, 3, 4, 5
    )
    src = src.transpose(2, 3)
    indices0 = torch.tensor([[0], [1]], dtype=torch.int64, device="cpu")
    indices1 = torch.tensor([[2, 1, 0]], dtype=torch.int64, device="cpu")
    mask = (
        torch.arange(2 * 3 * 5 * 4, device="cpu").reshape(2, 3, 5, 4).remainder(2).eq(0)
    )

    out = _unsafe_masked_index(src, mask, [indices0, indices1], fill=-3.0)
    ref = torch._unsafe_masked_index(src, mask, [indices0, indices1], fill=-3.0)

    torch.testing.assert_close(out, ref)


def test_unsafe_masked_index_does_not_dereference_masked_offsets():
    src = torch.arange(3, dtype=torch.float32, device="cpu")
    mask = torch.tensor([False], dtype=torch.bool, device="cpu")
    indices = torch.tensor([3], dtype=torch.int64, device="cpu")

    out = _unsafe_masked_index(src, mask, [indices], fill=-1.0)
    ref = torch._unsafe_masked_index(src, mask, [indices], fill=-1.0)

    torch.testing.assert_close(out, ref)
