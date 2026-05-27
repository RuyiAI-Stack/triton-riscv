import pytest
import torch

from .linspace import linspace


@pytest.mark.parametrize("steps", [1, 10, 128, 512, 1023, 1024])
def test_linspace_default(steps):
    tri_out = linspace(0, 1, steps, dtype=torch.float32, device="cpu")
    ref_out = torch.linspace(0, 1, steps, dtype=torch.float32, device="cpu")

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("steps", [5, 16, 100])
def test_linspace_negative_range(steps):
    tri_out = linspace(10, -10, steps, dtype=torch.float32, device="cpu")
    ref_out = torch.linspace(10, -10, steps, dtype=torch.float32, device="cpu")

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
