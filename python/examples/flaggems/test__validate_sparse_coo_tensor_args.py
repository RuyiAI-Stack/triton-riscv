import torch

from ._validate_sparse_coo_tensor_args import _validate_sparse_coo_tensor_args


def test__validate_sparse_coo_tensor_args():
    indices = torch.tensor([[0, 1], [0, 1]])
    assert _validate_sparse_coo_tensor_args(indices, torch.ones(2), (2, 2))
