"""Boundary regressions for the B13 CPU port."""

import importlib

import pytest
import torch
import torch.nn.functional as F


def op(name):
    package = importlib.import_module(__package__)
    assert name in package.__all__
    return getattr(package, name)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align", [False, True])
def test_grid_sample_boundaries(d, mode, padding, align):
    torch.manual_seed(44)
    x = torch.randn((1, 2) + ((3, 4) if d == 2 else (2, 3, 4))).transpose(-1, -2)
    grid = torch.rand((1,) + ((5, 7) if d == 2 else (3, 5, 7)) + (d,)) * 4 - 2
    grid.reshape(-1, d)[0] = 0
    ref = F.grid_sample(x, grid, mode=mode, padding_mode=padding, align_corners=align)
    actual = op("grid_sample")(x, grid, mode, padding, align)
    torch.testing.assert_close(actual, ref, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [2, 3])
def test_nearest_exact_noncontiguous(dtype, d):
    x = (
        torch.randn((1, 2) + ((3, 5) if d == 2 else (2, 3, 5)))
        .to(dtype)
        .transpose(-1, -2)
    )
    scales = (1.7, 0.7) if d == 2 else (1.3, 1.7, 0.7)
    size = tuple(int(a * b) for a, b in zip(x.shape[2:], scales))
    fn = op(f"_upsample_nearest_exact{d}d")
    torch.testing.assert_close(
        fn(x, size, *scales),
        F.interpolate(x, scale_factor=scales, mode="nearest-exact"),
    )


@pytest.mark.parametrize(
    "shape,size",
    [((1, 2, 9, 11), (3, 4)), ((1, 1, 2, 3), (1, 1)), ((1, 2, 5, 7), (8, 9))],
)
@pytest.mark.parametrize("align", [False, True])
def test_antialias_resize(shape, size, align):
    x = torch.randn(shape)
    torch.testing.assert_close(
        op("_upsample_bilinear2d_aa")(x, size, align),
        F.interpolate(x, size, mode="bilinear", align_corners=align, antialias=True),
        atol=3e-5,
        rtol=3e-5,
    )


@pytest.mark.parametrize("name", ["put", "put_"])
def test_put_rejects_bad_indices(name):
    with pytest.raises(IndexError):
        op(name)(torch.ones(3), torch.tensor([3]), torch.ones(1))
