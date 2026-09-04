import torch
import torch.nn.functional as F

from ._upsample_nearest_exact2d_backward import (
    _upsample_nearest_exact2d_backward,
)


def test_upsample_nearest_exact2d_backward():
    x = (
        torch.arange(4, dtype=torch.float32, device="cpu")
        .reshape(1, 1, 2, 2)
        .requires_grad_(True)
    )
    out = F.interpolate(x, size=(4, 4), mode="nearest-exact")
    grad = torch.ones_like(out)
    out.backward(grad)

    tri = _upsample_nearest_exact2d_backward(grad, (4, 4), (1, 1, 2, 2))

    torch.testing.assert_close(tri, x.grad)


def test_upsample_nearest_exact2d_backward_matches_center_based_indices():
    x = (
        torch.arange(15, dtype=torch.float32, device="cpu")
        .reshape(1, 1, 3, 5)
        .requires_grad_(True)
    )
    out = F.interpolate(x, size=(2, 4), mode="nearest-exact")
    grad = torch.arange(out.numel(), dtype=torch.float32, device="cpu").reshape_as(out)
    out.backward(grad)

    tri = _upsample_nearest_exact2d_backward(grad, (2, 4), (1, 1, 3, 5))

    torch.testing.assert_close(tri, x.grad)
