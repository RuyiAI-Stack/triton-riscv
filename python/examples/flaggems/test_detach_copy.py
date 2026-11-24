import torch

from .detach_copy import detach_copy


def test_detach_copy():
    torch.manual_seed(0)
    x = torch.randn(4)
    torch.testing.assert_close(detach_copy(x), x.detach().clone(), rtol=1e-4, atol=1e-4)
