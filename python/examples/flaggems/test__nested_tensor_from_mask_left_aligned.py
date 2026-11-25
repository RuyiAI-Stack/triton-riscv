import torch

from ._nested_tensor_from_mask_left_aligned import _nested_tensor_from_mask_left_aligned


def test__nested_tensor_from_mask_left_aligned():
    values = torch.arange(4.0)
    mask = torch.tensor([True, False, True, False])
    torch.testing.assert_close(
        _nested_tensor_from_mask_left_aligned(values, mask),
        torch.tensor([0.0, 2.0]),
        rtol=1e-4,
        atol=1e-4,
    )
