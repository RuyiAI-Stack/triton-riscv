import pytest
import torch

from .heaviside_ import heaviside_


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_heaviside_inplace(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    x[::7] = 0.0

    ref = torch.heaviside(x, torch.tensor(0.5))

    tri = x.clone()
    out = heaviside_(tri, 0.5)

    assert out is tri, "heaviside_ should return the same tensor"
    torch.testing.assert_close(tri, ref)


def test_heaviside_inplace_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cpu", dtype=torch.float32)
    x_sliced = x[::2, ::2]  # non-contiguous
    x_sliced[::7] = 0.0

    ref = torch.heaviside(x_sliced, torch.tensor(0.5))

    tri = x_sliced.clone()
    out = heaviside_(tri, 0.5)

    assert out is tri
    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
    ],
)
def test_heaviside__broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    ref_src = torch.randn(shape_a)
    a = ref_src
    b = torch.rand(shape_b)
    ref = a.clone()
    ref.heaviside_(b)

    tri = a.clone()
    out = heaviside_(tri, b)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_heaviside_inplace_tensor_values():
    torch.manual_seed(0)
    x = torch.randn(1024, device="cpu", dtype=torch.float32)
    x[::7] = 0.0
    values = torch.rand(1024, device="cpu", dtype=torch.float32)

    ref = torch.heaviside(x, values)

    tri = x.clone()
    out = heaviside_(tri, values)

    assert out is tri
    torch.testing.assert_close(tri, ref)


def test_heaviside_bf16():
    # tl.where miscompiles on bf16 (issue #119); the wrapper falls back to
    # torch.heaviside(out=x), so bf16 must match torch exactly.
    torch.manual_seed(0)
    x = torch.randn(128, device="cpu", dtype=torch.bfloat16)
    x[::7] = 0.0

    ref = torch.heaviside(x, torch.tensor(0.5, dtype=torch.bfloat16))

    tri = x.clone()
    out = heaviside_(tri, 0.5)

    assert out is tri
    torch.testing.assert_close(tri, ref)
