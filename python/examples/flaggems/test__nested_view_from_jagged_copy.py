import torch

from ._nested_view_from_jagged_copy import _nested_view_from_jagged_copy


def test__nested_view_from_jagged_copy():
    values = torch.arange(3.0)
    offsets = torch.tensor([0, 3])
    copied, copied_offsets = _nested_view_from_jagged_copy(values, offsets)
    torch.testing.assert_close(copied, values, rtol=1e-4, atol=1e-4)
    assert torch.equal(copied_offsets, offsets)
