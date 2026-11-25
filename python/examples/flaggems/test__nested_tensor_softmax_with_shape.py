import torch

from ._nested_tensor_softmax_with_shape import _nested_tensor_softmax_with_shape


def test__nested_tensor_softmax_with_shape():
    values = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(
        _nested_tensor_softmax_with_shape(values, None),
        torch.softmax(values, dim=-1),
        rtol=1e-4,
        atol=1e-4,
    )
