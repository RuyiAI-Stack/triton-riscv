import torch

from .mode import mode


def test_mode():
    x = torch.tensor([[1, 2, 2, 3], [4, 4, 5, 5]], dtype=torch.int64, device="cpu")

    out = mode(x, dim=1, keepdim=True)
    ref = torch.mode(x, dim=1, keepdim=True)

    torch.testing.assert_close(out.values, ref.values)
    torch.testing.assert_close(out.indices, ref.indices)
