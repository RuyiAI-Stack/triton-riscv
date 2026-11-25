import torch

from .linalg_vander import linalg_vander


def test_linalg_vander():
    values = torch.tensor([1.0, 2.0, -1.0, 0.0])
    torch.testing.assert_close(
        linalg_vander(values), torch.linalg.vander(values), rtol=1e-4, atol=1e-4
    )
