import torch

from ._validate_sparse_csc_tensor_args import _validate_sparse_csc_tensor_args


def test__validate_sparse_csc_tensor_args():
    tensor = torch.eye(2).to_sparse_csc()
    assert _validate_sparse_csc_tensor_args(
        tensor.ccol_indices(), tensor.row_indices(), tensor.values(), (2, 2)
    )
