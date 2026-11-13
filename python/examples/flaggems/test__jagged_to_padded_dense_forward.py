import torch

from ._jagged_to_padded_dense_forward import _jagged_to_padded_dense_forward


def test_jagged_to_padded_dense_forward():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu")
    offsets = torch.tensor([0, 0, 2, 5], dtype=torch.int64, device="cpu")

    out = _jagged_to_padded_dense_forward(values, [offsets], [4], padding_value=-1.0)
    ref = torch.ops.aten._jagged_to_padded_dense_forward(
        values, [offsets], [4], padding_value=-1.0
    )

    torch.testing.assert_close(out, ref)


def test_jagged_to_padded_dense_forward_truncates_to_max_length():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu")
    offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device="cpu")

    out = _jagged_to_padded_dense_forward(values, [offsets], [2], padding_value=-1.0)
    ref = torch.ops.aten._jagged_to_padded_dense_forward(
        values, [offsets], [2], padding_value=-1.0
    )

    torch.testing.assert_close(out, ref)
