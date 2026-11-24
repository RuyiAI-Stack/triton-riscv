import pytest
import torch

from .addcmul import addcmul, addcmul_out


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("value", [1.0, 2.5])
def test_addcmul(shape, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    t1 = torch.randn(shape, dtype=torch.float32, device="cpu")
    t2 = torch.randn(shape, dtype=torch.float32, device="cpu")

    ref_out = torch.addcmul(x, t1, t2, value=value)
    tri_out = addcmul(x, t1, t2, value=value)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 256)])
@pytest.mark.parametrize("value", [2.5])
def test_addcmul_out(shape, value):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    t1 = torch.randn(shape, dtype=torch.float32, device="cpu")
    t2 = torch.randn(shape, dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)

    ref_out = torch.addcmul(x, t1, t2, value=value)
    addcmul_out(x, t1, t2, value=value, out=out)

    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "input_shape, tensor1_shape, tensor2_shape",
    [((16, 1), (1, 257), (16, 257)), ((1,), (1023,), (1023,))],
)
@pytest.mark.parametrize("value", [1.0, 2.5])
def test_addcmul_broadcast(input_shape, tensor1_shape, tensor2_shape, value):
    torch.manual_seed(0)
    inp = torch.randn(input_shape, dtype=torch.float32, device="cpu")
    tensor1 = torch.randn(tensor1_shape, dtype=torch.float32, device="cpu")
    tensor2 = torch.randn(tensor2_shape, dtype=torch.float32, device="cpu")

    ref_out = torch.addcmul(inp, tensor1, tensor2, value=value)
    tri_out = addcmul(inp, tensor1, tensor2, value=value)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_addcmul_out_non_contiguous():
    torch.manual_seed(0)
    inp = torch.randn((32, 32), dtype=torch.float32, device="cpu")
    tensor1 = torch.randn((32, 32), dtype=torch.float32, device="cpu")
    tensor2 = torch.randn((32, 32), dtype=torch.float32, device="cpu")
    out_base = torch.empty((32, 64), dtype=torch.float32, device="cpu")
    out = out_base[:, ::2]

    ref_out = torch.addcmul(inp, tensor1, tensor2, value=2.5)
    ret = addcmul_out(inp, tensor1, tensor2, value=2.5, out=out)

    assert ret is out
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)


def test_addcmul_out_resizes_for_broadcast_result():
    torch.manual_seed(0)
    inp = torch.randn((16, 1), dtype=torch.float32, device="cpu")
    tensor1 = torch.randn((1, 257), dtype=torch.float32, device="cpu")
    tensor2 = torch.randn((16, 257), dtype=torch.float32, device="cpu")
    out = torch.empty((1,), dtype=torch.float32, device="cpu")

    ref_out = torch.addcmul(inp, tensor1, tensor2, value=2.5)
    ret = addcmul_out(inp, tensor1, tensor2, value=2.5, out=out)

    assert ret is out
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
