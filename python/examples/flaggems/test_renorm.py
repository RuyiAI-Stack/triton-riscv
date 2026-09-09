import torch

from .renorm import renorm, renorm_


def test_renorm_dim0():
    x = torch.tensor([[3.0, 4.0], [6.0, 8.0]], device="cpu")

    out = renorm(x, p=2.0, dim=0, maxnorm=5.0)
    ref = torch.renorm(x, p=2.0, dim=0, maxnorm=5.0)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_renorm_inplace():
    x = torch.tensor([[3.0, 4.0], [6.0, 8.0]], device="cpu")
    ref = x.clone().renorm_(p=2.0, dim=0, maxnorm=5.0)

    ret = renorm_(x, p=2.0, dim=0, maxnorm=5.0)

    assert ret is x
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)


def test_renorm_inplace_noncontiguous_dim0():
    x = torch.tensor([[3.0, 6.0], [4.0, 8.0]], device="cpu").transpose(0, 1)
    ref = x.clone().renorm_(p=2.0, dim=0, maxnorm=5.0)

    ret = renorm_(x, p=2.0, dim=0, maxnorm=5.0)

    assert ret is x
    assert not x.is_contiguous()
    torch.testing.assert_close(x, ref, rtol=1e-4, atol=1e-4)
