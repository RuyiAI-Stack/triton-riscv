import pytest
import torch

from .logical_xor_ import logical_xor_


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_logical_xor_inplace(shape, dtype):
    torch.manual_seed(0)
    x = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    y = torch.randint(0, 2, shape, dtype=dtype, device="cpu")
    ref = x.clone()

    ref.logical_xor_(y)
    ret = logical_xor_(x, y)

    assert ret is x
    torch.testing.assert_close(x, ref)


def test_logical_xor_inplace_broadcast():
    torch.manual_seed(0)
    x = torch.randint(0, 2, (16, 257), dtype=torch.bool, device="cpu")
    y = torch.randint(0, 2, (1, 257), dtype=torch.bool, device="cpu")
    ref = x.clone()

    ref.logical_xor_(y)
    ret = logical_xor_(x, y)

    assert ret is x
    torch.testing.assert_close(x, ref)


def test_logical_xor_inplace_non_contiguous():
    torch.manual_seed(0)
    x_base = torch.randint(0, 2, (32, 64), dtype=torch.bool, device="cpu")
    x = x_base[:, ::2]
    y = torch.randint(0, 2, x.shape, dtype=torch.bool, device="cpu")
    ref = x.clone()

    ref.logical_xor_(y)
    ret = logical_xor_(x, y)

    assert ret is x
    torch.testing.assert_close(x, ref)


def test_logical_xor_inplace_rejects_invalid_same_numel_shape():
    x = torch.ones((2, 2), dtype=torch.bool, device="cpu")
    y = torch.ones((1, 4), dtype=torch.bool, device="cpu")

    with pytest.raises(RuntimeError):
        x.clone().logical_xor_(y)
    with pytest.raises(RuntimeError):
        logical_xor_(x.clone(), y)


def test_logical_xor_inplace_empty_tensor():
    x = torch.empty((0,), dtype=torch.bool, device="cpu")
    y = torch.empty((0,), dtype=torch.bool, device="cpu")
    ref = x.clone()

    ret = logical_xor_(x, y)
    ref.logical_xor_(y)

    assert ret is x
    torch.testing.assert_close(x, ref)


def test_logical_xor_inplace_scalar_tensor():
    x = torch.tensor([True, False, True, False], dtype=torch.bool, device="cpu")
    y = torch.tensor(False, dtype=torch.bool, device="cpu")
    ref = x.clone()

    ret = logical_xor_(x, y)
    ref.logical_xor_(y)

    assert ret is x
    torch.testing.assert_close(x, ref)
