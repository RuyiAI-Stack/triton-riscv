import pytest
import torch

from .nextafter import nextafter


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_nextafter(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32)
    y = torch.randn(size, device="cpu", dtype=torch.float32)

    ref = torch.nextafter(x, y)
    tri = nextafter(x, y)

    # nextafter is uniquely defined by IEEE 754, so the results are expected
    # to be bit-identical, not merely close.
    assert torch.equal(tri, ref)


def test_nextafter_boundaries():
    x = torch.tensor(
        [0.0, 0.0, 1.0, -1.0, 0.5, -0.0, -0.0, 0.0, 0.0], dtype=torch.float32
    )
    y = torch.tensor(
        [1.0, -1.0, 2.0, -2.0, 0.5, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32
    )

    ref = torch.nextafter(x, y)
    tri = nextafter(x, y)

    assert torch.equal(tri, ref)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_nextafter_fallback(dtype):
    # The non-fp32 dtypes delegate to torch; this test pins that fallback.
    x = torch.randn(128, device="cpu", dtype=torch.float32).to(dtype)
    y = torch.randn(128, device="cpu", dtype=torch.float32).to(dtype)

    ref = torch.nextafter(x, y)
    tri = nextafter(x, y)

    assert torch.equal(tri, ref)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((16, 256), (1, 256)),
        ((4, 128), (4, 1)),
        ((256,), (16, 256)),
    ],
)
def test_nextafter_broadcast(shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randn(shape_a)
    b = torch.randn(shape_b)

    ref_out = torch.nextafter(a, b)
    tri_out = nextafter(a, b)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_nextafter_nan_bits():
    # torch.nextafter quiets NaN into 0x7FC00000; the kernel must match
    # bit-for-bit (emitting the sNaN pattern 0x7F800001 would be a hazard).
    x = torch.tensor([1.0, float("nan"), 2.0, 3.0])
    y = torch.tensor([float("nan"), 1.0, float("nan"), 3.0])

    ref = torch.nextafter(x, y)
    tri = nextafter(x, y)

    assert torch.equal(tri.view(torch.int32), ref.view(torch.int32))


def test_nextafter_equal_signed_zeros():
    # torch returns `other` when x == other: +0 for (-0, +0) and -0 for
    # (+0, -0). Compare bit patterns because 0.0 == -0.0.
    for a, b, want in (
        (-0.0, 0.0, 0.0),
        (0.0, -0.0, -0.0),
    ):
        got = nextafter(torch.tensor(a), torch.tensor(b)).item()
        assert got == 0.0 and str(got)[0] == str(want)[0] or got == want, (a, b, got)
        ref = torch.nextafter(torch.tensor(a), torch.tensor(b)).item()
        import struct

        assert struct.pack("<d", got) == struct.pack("<d", ref)
