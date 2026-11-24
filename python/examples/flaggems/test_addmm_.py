import torch

from .addmm_ import addmm_


def test_addmm_():
    torch.manual_seed(0)
    self = torch.randn(3, 4, dtype=torch.float32, device="cpu")
    mat1 = torch.randn(3, 5, dtype=torch.float32, device="cpu")
    mat2 = torch.randn(5, 4, dtype=torch.float32, device="cpu")
    ref = self.clone()

    ref.addmm_(mat1, mat2, beta=0.5, alpha=2.0)
    out = addmm_(self, mat1, mat2, beta=0.5, alpha=2.0)

    assert out is self
    torch.testing.assert_close(self, ref, rtol=1e-4, atol=1e-4)
