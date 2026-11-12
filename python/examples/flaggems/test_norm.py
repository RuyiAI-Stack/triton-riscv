import pytest
import torch

from .norm import norm

BF16_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="backend: fp32 to bfloat16 conversion miscompiles (RuyiAI-Stack/triton-riscv#114)",
)


@pytest.mark.parametrize("p", [2, 1, 3, 0.5, -1, -0.5, float("inf"), float("-inf"), 0])
@pytest.mark.parametrize("size", [512, 1023, 37])
def test_norm_global(p, size):
    torch.manual_seed(0)
    x = torch.randn(size, device="cpu", dtype=torch.float32) * 3.0

    ref = torch.linalg.norm(x, ord=p)
    tri = norm(x, p=p)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("p", [2, 1, 3, float("inf"), float("-inf"), 0])
def test_norm_dim(p):
    torch.manual_seed(0)
    x = torch.randn(16, 256, device="cpu", dtype=torch.float32) * 3.0

    ref = torch.linalg.norm(x, ord=p, dim=1)
    tri = norm(x, p=p, dim=1)

    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


def test_norm_keepdim():
    torch.manual_seed(0)
    x = torch.randn(16, 256, device="cpu", dtype=torch.float32) * 3.0

    ref = torch.linalg.norm(x, ord=2, dim=1, keepdim=True)
    tri = norm(x, p=2, dim=1, keepdim=True)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_norm_fp16(dtype):
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 3.0).to(dtype)

    ref = torch.linalg.norm(x.float(), ord=2).to(dtype)
    tri = norm(x, p=2)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_norm_fp64_fallback():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float64) * 3.0

    ref = torch.linalg.norm(x, ord=2)
    tri = norm(x, p=2)

    torch.testing.assert_close(tri, ref)


@BF16_XFAIL
def test_norm_bf16():
    torch.manual_seed(0)
    x = (torch.randn(128, dtype=torch.float32) * 3.0).to(torch.bfloat16)

    ref = torch.linalg.norm(x.float(), ord=2).to(torch.bfloat16)
    tri = norm(x, p=2)

    torch.testing.assert_close(tri.float(), ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("p", [-1, -0.5])
def test_norm_negative_p_dim_tail(p):
    # N=17 forces padding lanes inside a BLOCK_N row window; a negative ord
    # would turn the padding sentinel into inf and poison the row sum.
    x = torch.randn(16, 17)
    torch.testing.assert_close(
        norm(x, p=p, dim=1), torch.linalg.norm(x, ord=p, dim=1), rtol=1e-4, atol=1e-4
    )


def test_norm_empty_tensor():
    torch.testing.assert_close(norm(torch.empty(0)), torch.linalg.norm(torch.empty(0)))


def test_norm_integer_raises_like_torch():
    with pytest.raises(NotImplementedError):
        norm(torch.tensor([3, 4]))
