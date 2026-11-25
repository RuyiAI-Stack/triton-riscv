import torch

from .copy_sparse_to_sparse_ import copy_sparse_to_sparse_


def test_copy_sparse_to_sparse_():
    src = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse()
    dst = torch.zeros_like(src)
    copy_sparse_to_sparse_(dst, src)
    torch.testing.assert_close(dst.to_dense(), src.to_dense(), rtol=1e-4, atol=1e-4)
