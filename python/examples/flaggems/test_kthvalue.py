import torch

from .kthvalue import kthvalue


def test_kthvalue():
    x = torch.tensor([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]], device="cpu")

    values, indices = kthvalue(x, 2, dim=1, keepdim=True)
    ref_values, ref_indices = torch.kthvalue(x, 2, dim=1, keepdim=True)

    torch.testing.assert_close(values, ref_values)
    torch.testing.assert_close(indices, ref_indices)


def test_kthvalue_preserves_float64_precision():
    x = torch.tensor(
        [[float(2**24) - 0.25, float(2**24) - 1.25, float(2**24) - 2.25]],
        dtype=torch.float64,
        device="cpu",
    )

    values, indices = kthvalue(x, 2, dim=1, keepdim=False)
    ref_values, ref_indices = torch.kthvalue(x, 2, dim=1, keepdim=False)

    torch.testing.assert_close(values, ref_values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(indices, ref_indices)


def test_kthvalue_preserves_int64_values():
    x = torch.tensor(
        [[2**40 + 3, 2**40 + 1, 2**40 + 2]],
        dtype=torch.int64,
        device="cpu",
    )

    values, indices = kthvalue(x, 2, dim=1, keepdim=True)
    ref_values, ref_indices = torch.kthvalue(x, 2, dim=1, keepdim=True)

    assert values.dtype == torch.int64
    torch.testing.assert_close(values, ref_values)
    torch.testing.assert_close(indices, ref_indices)


def test_kthvalue_ties_match_torch_values_and_return_valid_indices():
    x = torch.tensor([[2.0, 1.0, 1.0, 2.0], [3.0, 3.0, 1.0, 1.0]], device="cpu")

    values, indices = kthvalue(x, 2, dim=1, keepdim=True)
    ref_values, _ = torch.kthvalue(x, 2, dim=1, keepdim=True)

    # PyTorch does not guarantee which index is selected for tied values.
    torch.testing.assert_close(values, ref_values)
    torch.testing.assert_close(x.gather(1, indices), values)
