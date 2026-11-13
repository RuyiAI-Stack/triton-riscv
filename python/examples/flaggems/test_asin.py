import torch

from .asin import asin, asin_


def test_asin_aliases_arcsin():
    x = torch.linspace(-0.75, 0.75, 1024, dtype=torch.float32, device="cpu")
    ref = torch.asin(x)

    out = asin(x)
    x_inplace = x.clone()
    asin_(x_inplace)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_inplace, ref, rtol=1e-5, atol=1e-5)
