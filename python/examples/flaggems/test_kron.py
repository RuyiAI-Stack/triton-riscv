import pytest
import torch

from .kron import kron


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((512,), (512,)),
        ((1023,), (1023,)),
        ((1024,), (1024,)),
        ((2, 512), (2, 512)),
        ((2, 1023), (2, 1023)),
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64]
)
def test_kron(shape1, shape2, dtype):
    a = torch.randn(shape1, dtype=dtype)
    b = torch.randn(shape2, dtype=dtype)

    # Requires requires_grad=False because the backward is not implemented

    out_triton = kron(a, b)
    out_torch = torch.kron(a, b)

    torch.testing.assert_close(out_triton, out_torch)
