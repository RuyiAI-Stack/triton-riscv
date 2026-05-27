import pytest
import torch

from .nan_to_num import nan_to_num


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_nan_to_num(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    # Inject some NaN and Inf values
    x.view(-1)[0] = float("nan")
    x.view(-1)[1] = float("inf")
    x.view(-1)[2] = -float("inf")

    ref_out = torch.nan_to_num(x)
    tri_out = nan_to_num(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_nan_to_num_custom(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    x.view(-1)[0] = float("nan")
    x.view(-1)[1] = float("inf")
    x.view(-1)[2] = -float("inf")

    ref_out = torch.nan_to_num(x, nan=1.0, posinf=100.0, neginf=-100.0)
    tri_out = nan_to_num(x, nan=1.0, posinf=100.0, neginf=-100.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
