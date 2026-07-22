import pytest
import torch

from ._amp_foreach_non_finite_check_and_unscale_ import (
    _amp_foreach_non_finite_check_and_unscale_,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("contains_non_finite", [False, True])
def test_amp_foreach_non_finite_check_and_unscale_(dtype, contains_non_finite):
    values = torch.linspace(-2.0, 2.0, 1023, dtype=dtype, device="cpu")
    if contains_non_finite:
        values[31] = float("inf")
        values[997] = float("nan")
    tensors = [values.clone(), values.flip(0).clone()]
    ref_tensors = [tensor.clone() for tensor in tensors]
    found_inf = torch.zeros((), dtype=torch.float32, device="cpu")
    ref_found_inf = torch.zeros((), dtype=torch.float32, device="cpu")
    inv_scale = torch.tensor(0.5, dtype=torch.float32, device="cpu")

    _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale)
    torch._amp_foreach_non_finite_check_and_unscale_(
        ref_tensors, ref_found_inf, inv_scale
    )

    for actual, expected in zip(tensors, ref_tensors):
        torch.testing.assert_close(actual, expected, equal_nan=True)
    torch.testing.assert_close(found_inf, ref_found_inf)


def test_amp_foreach_non_finite_check_and_unscale_non_contiguous():
    base0 = torch.arange(12, dtype=torch.float32, device="cpu").reshape(3, 4)
    base1 = torch.arange(12, 24, dtype=torch.float32, device="cpu").reshape(3, 4)
    tensors = [base0.t(), base1[:, ::2]]
    ref_tensors = [
        torch.empty_strided(
            tensor.size(), tensor.stride(), dtype=tensor.dtype, device=tensor.device
        ).copy_(tensor)
        for tensor in tensors
    ]
    found_inf = torch.zeros((), dtype=torch.float32, device="cpu")
    ref_found_inf = torch.zeros((), dtype=torch.float32, device="cpu")
    inv_scale = torch.tensor(0.25, dtype=torch.float32, device="cpu")

    _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale)
    torch._amp_foreach_non_finite_check_and_unscale_(
        ref_tensors, ref_found_inf, inv_scale
    )

    for actual, expected in zip(tensors, ref_tensors):
        assert not actual.is_contiguous()
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(found_inf, ref_found_inf)


def test_amp_foreach_non_finite_check_and_unscale_preserves_existing_found_inf():
    tensors = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cpu")]
    found_inf = torch.ones((), dtype=torch.float32, device="cpu")
    inv_scale = torch.tensor(0.5, dtype=torch.float32, device="cpu")

    _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale)

    torch.testing.assert_close(
        tensors[0], torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
    )
    torch.testing.assert_close(found_inf, torch.ones((), dtype=torch.float32))


def test_amp_foreach_non_finite_check_and_unscale_non_contiguous_non_finite():
    base = torch.arange(16, dtype=torch.float32, device="cpu").reshape(4, 4)
    tensor = base[:, ::2]
    tensor[1, 1] = float("inf")
    ref_tensor = torch.empty_strided(
        tensor.size(), tensor.stride(), dtype=tensor.dtype, device=tensor.device
    ).copy_(tensor)
    found_inf = torch.zeros((), dtype=torch.float32, device="cpu")
    ref_found_inf = torch.zeros((), dtype=torch.float32, device="cpu")
    inv_scale = torch.tensor(0.25, dtype=torch.float32, device="cpu")

    _amp_foreach_non_finite_check_and_unscale_([tensor], found_inf, inv_scale)
    torch._amp_foreach_non_finite_check_and_unscale_(
        [ref_tensor], ref_found_inf, inv_scale
    )

    assert not tensor.is_contiguous()
    torch.testing.assert_close(tensor, ref_tensor, equal_nan=True)
    torch.testing.assert_close(found_inf, ref_found_inf)


@pytest.mark.parametrize("dtype", [torch.int32, torch.complex64])
def test_amp_foreach_non_finite_check_and_unscale_rejects_unsupported_dtype(dtype):
    tensors = [torch.ones(2, dtype=dtype, device="cpu")]
    found_inf = torch.zeros((), dtype=torch.float32, device="cpu")
    inv_scale = torch.tensor(0.5, dtype=torch.float32, device="cpu")

    with pytest.raises(NotImplementedError):
        _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale)
