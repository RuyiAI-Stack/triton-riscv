import pytest
import torch

from .prelu import prelu


def _prelu_reference(a, w):
    if w.numel() == 1:
        alpha = w.reshape(())
    elif a.dim() == 1:
        alpha = w
    else:
        alpha_shape = [1] * a.dim()
        alpha_shape[1] = w.numel()
        alpha = w.reshape(alpha_shape)
    return torch.where(a >= 0, a, alpha * a)


@pytest.mark.parametrize(
    "shape, num_channels",
    [
        ((512,), 1),
        ((512,), 512),
        ((2, 512), 1),
        ((2, 512), 512),
        ((2, 3, 32, 32), 1),
        ((2, 3, 32, 32), 3),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_prelu(shape, num_channels, dtype):
    a = torch.randn(shape, dtype=dtype)
    w = torch.randn((num_channels,), dtype=dtype)

    out_triton = prelu(a, w)
    if a.dim() == 1 and w.numel() != 1:
        # torch.nn.functional.prelu rejects this migrated per-element 1D case.
        out_torch = _prelu_reference(a, w)
    else:
        out_torch = torch.nn.functional.prelu(a, w)

    torch.testing.assert_close(out_triton, out_torch)
