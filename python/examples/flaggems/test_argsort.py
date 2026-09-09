import pytest
import torch

from .argsort import argsort


@pytest.mark.parametrize("descending", [False, True])
def test_argsort(descending):
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]], device="cpu")

    out = argsort(x, dim=-1, descending=descending)
    ref = torch.argsort(x, dim=-1, descending=descending)

    torch.testing.assert_close(out, ref)
