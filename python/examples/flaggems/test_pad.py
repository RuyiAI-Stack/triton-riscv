import pytest
import torch
import torch.nn.functional as F

from .pad import constant_pad_nd, pad


@pytest.mark.parametrize(
    "shape, pad_list",
    [
        ((4, 3, 16, 16), [2, 2, 2, 2]),
        ((2, 3, 8, 8), [1, 2, 3, 4]),
        ((4, 3, 16, 16), [0, 0, 0, 0]),
        ((4, 3, 15, 17), [2, 2, 2, 2]),
        ((4, 3, 512, 64), [1, 1, 1, 1]),
        ((4, 3, 1023, 64), [1, 1, 1, 1]),
        ((4, 3, 1024, 64), [1, 1, 1, 1]),
    ],
)
def test_pad_constant(shape, pad_list):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = F.pad(x, pad_list, mode="constant", value=0.0)
    tri_out = pad(x, pad_list, mode="constant", value=0.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, pad_list",
    [
        ((4, 3, 16, 16), [2, 2, 2, 2]),
        ((2, 3, 12, 10), [1, 2, 1, 2]),
    ],
)
def test_pad_constant_value(shape, pad_list):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = F.pad(x, pad_list, mode="constant", value=3.14)
    tri_out = pad(x, pad_list, mode="constant", value=3.14)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, pad_list",
    [
        ((8, 3, 512, 512), [1, 1, 1, 1]),
    ],
)
def test_pad_constant_large(shape, pad_list):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = F.pad(x, pad_list, mode="constant", value=0.0)
    tri_out = pad(x, pad_list, mode="constant", value=0.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, pad_list",
    [
        ((4, 3, 16, 16), [2, 2, 2, 2]),
        ((4, 3, 10, 12), [1, 2, 3, 4]),
    ],
)
def test_pad_3d(shape, pad_list):
    torch.manual_seed(0)
    # Only pad H, W dimensions
    pad_list_hw = pad_list[-4:]
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = F.pad(x, pad_list_hw, mode="constant", value=0.0)
    tri_out = pad(x, pad_list_hw, mode="constant", value=0.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, pad_list",
    [
        ((4, 3, 16, 16), [2, 2, 2, 2]),
        ((2, 3, 8, 8), [1, 2, 3, 4]),
        ((4, 3, 16, 16), [0, 0, 0, 0]),
        ((4, 3, 15, 17), [2, 2, 2, 2]),
        ((4, 3, 512, 64), [1, 1, 1, 1]),
        ((4, 3, 1023, 64), [1, 1, 1, 1]),
        ((4, 3, 1024, 64), [1, 1, 1, 1]),
    ],
)
def test_constant_pad_nd(shape, pad_list):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = F.pad(x, pad_list, mode="constant", value=0.0)
    tri_out = constant_pad_nd(x, pad_list, value=0.0)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape, pad_list",
    [
        ((4, 3, 16, 16), [2, 2, 2, 2]),
        ((2, 3, 12, 10), [1, 2, 1, 2]),
    ],
)
def test_constant_pad_nd_nonzero_value(shape, pad_list):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = F.pad(x, pad_list, mode="constant", value=1.5)
    tri_out = constant_pad_nd(x, pad_list, value=1.5)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
