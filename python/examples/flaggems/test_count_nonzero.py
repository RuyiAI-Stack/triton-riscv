import pytest
import torch

from .count_nonzero import count_nonzero


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_count_nonzero(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    # Set some values to zero
    x[x.abs() < 0.5] = 0.0

    ref = torch.count_nonzero(x)
    tri = count_nonzero(x)

    assert tri == ref, (
        f"count_nonzero failed for shape {shape}: tri={tri}, ref={ref}"
    )


@pytest.mark.parametrize("shape", [(4, 128), (8, 256)])
def test_count_nonzero_dim(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x[x.abs() < 0.5] = 0.0

    for dim in [0, 1]:
        ref = torch.count_nonzero(x, dim=dim)
        tri = count_nonzero(x, dim=dim)
        torch.testing.assert_close(tri, ref, rtol=0, atol=0)
