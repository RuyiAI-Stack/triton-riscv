import pytest
import torch

from .amin import amin


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_amin_global(size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 5.0

    ref = torch.amin(x)
    tri = amin(x)

    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize(
    "shape, dim", [((16, 256), 1), ((16, 256), 0), ((4, 8, 32), 2), ((4, 8, 32), -1)]
)
def test_amin_dim(shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cpu", dtype=torch.float32) * 5.0

    ref = torch.amin(x, dim=dim)
    tri = amin(x, dim=dim)

    torch.testing.assert_close(tri, ref)


def test_amin_multi_dim():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32, device="cpu", dtype=torch.float32) * 5.0

    ref = torch.amin(x, dim=(0, 2))
    tri = amin(x, dim=(0, 2))

    torch.testing.assert_close(tri, ref)


def test_amin_keepdim():
    torch.manual_seed(0)
    x = torch.randn(16, 256, device="cpu", dtype=torch.float32) * 5.0

    ref = torch.amin(x, dim=1, keepdim=True)
    tri = amin(x, dim=1, keepdim=True)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref)


def test_amin_int():
    torch.manual_seed(0)
    x = torch.randint(-1000, 1001, (16, 256), dtype=torch.int32)

    ref = torch.amin(x, dim=1)
    tri = amin(x, dim=1)

    torch.testing.assert_close(tri, ref)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_amin_fp16(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 5.0).to(dtype)

    ref = torch.amin(x)
    tri = amin(x)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3)


BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@BF16_XFAIL
def test_amin_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 5.0).to(torch.bfloat16)

    ref = torch.amin(x)
    tri = amin(x)

    torch.testing.assert_close(tri, ref)


def test_amin_uint8():
    # All-positive unsigned data: the padding sentinel must be the dtype
    # maximum (2^8 - 1), not 0, or the min collapses to 0.
    torch.manual_seed(0)
    x = torch.randint(5, 250, (16, 256), dtype=torch.uint8)

    ref = torch.amin(x, dim=1)
    tri = amin(x, dim=1)

    torch.testing.assert_close(tri, ref)


def test_amin_uint8_global():
    x = torch.randint(5, 250, (1023,), dtype=torch.uint8)

    ref = torch.amin(x)
    tri = amin(x)

    torch.testing.assert_close(tri, ref)


def test_amin_bool():
    # The int1 dtype has no sentinel in get_dtype_max; pin the working path.
    x = torch.tensor([True, False, True])
    torch.testing.assert_close(amin(x), torch.amin(x))
    torch.testing.assert_close(
        amin(torch.ones(5, dtype=torch.bool)),
        torch.amin(torch.ones(5, dtype=torch.bool)),
    )


def test_amin_empty_and_zero_dim():
    with pytest.raises(IndexError):
        amin(torch.empty(3, 0), dim=1)  # torch: expected reduction dim non-zero
