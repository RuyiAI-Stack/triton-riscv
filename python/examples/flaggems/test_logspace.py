import math

import pytest
import torch

from .logspace import logspace


@pytest.mark.parametrize("steps", [512, 1023, 1024])
def test_logspace(steps):
    torch.manual_seed(0)
    start = 0.1
    end = 2.0

    ref_out = torch.logspace(start, end, steps=steps, base=10.0)
    tri_out = logspace(start, end, steps=steps, base=10.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("base", [2.0, math.e])
def test_logspace_base(base):
    steps = 1024
    start = 0.1
    end = 2.0

    ref_out = torch.logspace(start, end, steps=steps, base=base)
    tri_out = logspace(start, end, steps=steps, base=base)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
