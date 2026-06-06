import pytest
import torch

from .grid_sample import grid_sample


@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize(
    "N, C, H_in, W_in, H_out, W_out",
    [
        (2, 3, 8, 8, 10, 12),
        (1, 1, 8, 8, 16, 16),
    ],
)
def test_grid_sample_2d_nearest_zeros(N, C, H_in, W_in, H_out, W_out, align_corners):
    torch.manual_seed(0)
    input = torch.randn(N, C, H_in, W_in, device="cpu", dtype=torch.float32)
    grid = torch.rand(N, H_out, W_out, 2, device="cpu", dtype=torch.float32) * 2 - 1

    ref = torch.nn.functional.grid_sample(
        input,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    tri = grid_sample(
        input,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=align_corners,
    )

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sample_2d_bilinear_zeros(align_corners):
    torch.manual_seed(0)
    N, C, H_in, W_in, H_out, W_out = 1, 2, 8, 8, 10, 12
    input = torch.randn(N, C, H_in, W_in, device="cpu", dtype=torch.float32)
    grid = torch.rand(N, H_out, W_out, 2, device="cpu", dtype=torch.float32) * 2 - 1

    ref = torch.nn.functional.grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    tri = grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )

    if not align_corners:
        rtol, atol = 1e-2, 1.1
    else:
        rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)
