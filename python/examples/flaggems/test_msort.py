import torch

from .msort import msort


def test_msort():
    values = torch.tensor([[3.0, 1.0], [2.0, 0.0]])
    torch.testing.assert_close(msort(values), torch.msort(values), rtol=1e-4, atol=1e-4)
