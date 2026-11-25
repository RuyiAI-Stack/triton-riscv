"""Boundary regressions for the B14 CPU port."""

import importlib

import pytest
import torch
import torch.nn.functional as F


def op(name):
    package = importlib.import_module(__package__)
    assert name in package.__all__
    return getattr(package, name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pool_noncontiguous(dtype):
    x = torch.randn(1, 2, 5, 7, 9).to(dtype).transpose(-1, -2)
    torch.testing.assert_close(
        op("adaptive_max_pool3d")(x, (2, 3, 4)),
        F.adaptive_max_pool3d(x, (2, 3, 4), return_indices=True),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_grouped_convolution(dtype):
    x = torch.randn(1, 4, 7, 9).to(dtype)
    w = torch.randn(6, 2, 3, 3).to(dtype)
    b = torch.randn(6).to(dtype)
    args = (x, w, b, [2, 1], [1, 1], [1, 1], False, [0, 0], 2)
    torch.testing.assert_close(
        op("convolution")(*args),
        F.conv2d(x, w, b, stride=(2, 1), padding=1, groups=2),
        atol=(
            0.04
            if dtype == torch.bfloat16
            else 0.004
            if dtype == torch.float16
            else 3e-5
        ),
        rtol=0.02 if dtype != torch.float32 else 3e-5,
    )
