import pytest
import torch
import torch.nn.functional as F

from .affine_grid_generator import affine_grid_generator


@pytest.mark.parametrize("N, H, W", [(2, 4, 6), (1, 8, 8)])
@pytest.mark.parametrize("align_corners", [True, False])
def test_affine_grid_generator(N, H, W, align_corners):
    torch.manual_seed(0)
    theta = torch.randn(N, 2, 3, device="cpu", dtype=torch.float32)

    ref = F.affine_grid(theta, (N, 3, H, W), align_corners=align_corners)
    tri = affine_grid_generator(
        theta, (N, 3, H, W), align_corners=align_corners
    )

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("align_corners", [True, False])
def test_affine_grid_generator_identity(align_corners):
    N, H, W = 1, 10, 10
    theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    ref = F.affine_grid(theta, (N, 3, H, W), align_corners=align_corners)
    tri = affine_grid_generator(
        theta, (N, 3, H, W), align_corners=align_corners
    )

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_affine_grid_generator_shape_validation():
    theta = torch.randn(2, 2, 3)

    with pytest.raises(AssertionError):
        affine_grid_generator(theta, (2, 3), False)
