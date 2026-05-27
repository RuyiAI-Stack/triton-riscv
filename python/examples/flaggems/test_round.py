import pytest
import torch

from .round import round, round_, round_out


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_round(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.round(x)
    tri_out = round(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_round_decimals(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu") * 100

    ref_out = torch.round(x, decimals=1)
    tri_out = round(x, decimals=1)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_round_fp16(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float16, device="cpu")

    ref_out = torch.round(x)
    tri_out = round(x)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_round_inplace(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref = x.clone()
    ref_out = torch.round(ref)
    tri = x.clone()
    result = round_(tri)

    torch.testing.assert_close(tri, ref_out, rtol=1e-4, atol=1e-4)
    assert result is tri, "round_ must return the input tensor"


@pytest.mark.parametrize("shape", [(512,), (1024,)])
def test_round_out(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.round(x)
    out = torch.empty_like(x)
    result = round_out(x, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
    assert result is out, "round_out must return the output tensor"
