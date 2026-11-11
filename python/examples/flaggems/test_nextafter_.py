import pytest
import torch

from .nextafter_ import nextafter_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_nextafter_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)

    ref = torch.nextafter(x.clone(), y)

    tri = x.clone()
    out = nextafter_(tri, y)

    assert out is tri, "nextafter_ should return the same tensor"
    assert torch.equal(tri, ref)


def test_nextafter_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    y = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    x_sliced = x[::2, ::2]  # non-contiguous

    ref = torch.nextafter(x_sliced.clone(), y[::2, ::2])

    tri = x_sliced.clone()
    out = nextafter_(tri, y[::2, ::2])

    assert out is tri
    assert torch.equal(tri, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
    ],
)
def test_nextafter__broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    ref_src = torch.randn(shape_a)
    a = ref_src
    b = torch.randn(shape_b)
    ref = torch.nextafter(a, b)

    tri = a.clone()
    out = nextafter_(tri, b)

    assert out is tri
    assert torch.equal(tri, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_nextafter_fallback(dtype):
    # The non-fp32 dtypes delegate to torch; this test pins that fallback.
    x = (torch.randn(128, device="cpu", dtype=torch.float32)).to(dtype)
    y = (torch.randn(128, device="cpu", dtype=torch.float32)).to(dtype)

    ref = torch.nextafter(x, y)
    tri = nextafter_(x.clone(), y)

    assert torch.equal(tri, ref)
