import pytest
import torch

from .masked_scatter import masked_scatter, masked_scatter_


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128), (512,), (1023,), (1024,)],
)
def test_masked_scatter(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    num_true = mask.sum().item()
    source = torch.randn(num_true, dtype=torch.float32, device="cpu")

    ref_out = torch.masked_scatter(x, mask, source)
    tri_out = masked_scatter(x, mask, source)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "shape",
    [(16, 256), (4, 128)],
)
def test_masked_scatter_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu")
    num_true = mask.sum().item()
    source = torch.randn(num_true, dtype=torch.float32, device="cpu")
    x_ref = x.clone()

    x_ref.masked_scatter_(mask, source)
    masked_scatter_(x, mask, source)

    torch.testing.assert_close(x, x_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("inplace", [False, True])
def test_masked_scatter_rejects_short_source(inplace):
    x = torch.arange(6, dtype=torch.float32)
    original = x.clone()
    mask = torch.tensor([True, False, True, True, False, False])
    source = torch.tensor([10.0, 20.0])

    with pytest.raises(RuntimeError, match="source"):
        if inplace:
            masked_scatter_(x, mask, source)
        else:
            masked_scatter(x, mask, source)

    torch.testing.assert_close(x, original)
