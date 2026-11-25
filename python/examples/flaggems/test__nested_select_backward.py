import torch

from ._nested_select_backward import _nested_select_backward


def test__nested_select_backward():
    grad = torch.tensor([5.0])
    out = _nested_select_backward(grad, 1, (3,))
    torch.testing.assert_close(out, torch.tensor([0.0, 5.0, 0.0]), rtol=1e-4, atol=1e-4)
