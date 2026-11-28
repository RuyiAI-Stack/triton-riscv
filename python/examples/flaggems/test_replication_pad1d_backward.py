import importlib

import pytest
import torch
import torch.nn.functional as F

from .replication_pad1d_backward import replication_pad1d_backward


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((2, 5), (2, 2)),
        ((3, 6), (1, 2)),
        ((2, 3, 7), (3, 1)),
        ((2, 4), (0, 0)),
        ((2, 4), (0, 3)),
        ((2, 4), (2, 0)),
        ((1, 1), (2, 3)),  # W_in == 1: every output sums into the lone element
        ((2, 5), (-1, 2)),  # negative left pad crops column 0 (no gradient)
        ((2, 5), (2, -2)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_replication_pad1d_backward_matches_aten(shape, padding, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    w_out = shape[-1] + padding[0] + padding[1]
    grad = torch.randn(*shape[:-1], w_out, dtype=dtype)
    out = replication_pad1d_backward(grad, x, padding)
    ref = torch.ops.aten.replication_pad1d_backward(grad, x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_replication_pad1d_backward_low_precision_exact(dtype):
    torch.manual_seed(0)
    x = torch.randn(2, 5, dtype=dtype)
    grad = torch.randn(2, 9, dtype=dtype)
    out = replication_pad1d_backward(grad, x, (2, 2))
    ref = torch.ops.aten.replication_pad1d_backward(grad, x, (2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad1d_backward_complex_matches_aten():
    torch.manual_seed(0)
    x = torch.randn(2, 5, dtype=torch.complex128)
    grad = torch.randn(2, 9, dtype=torch.complex128)
    out = replication_pad1d_backward(grad, x, (2, 2))
    ref = torch.ops.aten.replication_pad1d_backward(grad, x, (2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad1d_backward_autograd_crosscheck():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(2, 3, 10, dtype=torch.float64)
    y = F.pad(x, (2, 2), mode="replicate")
    y.backward(grad)
    out = replication_pad1d_backward(grad, x.detach(), (2, 2))
    torch.testing.assert_close(out, x.grad, rtol=0, atol=0)


def test_replication_pad1d_backward_edge_sums_sequential():
    # Column 0 sums grad_out[0:3] in ascending order; column W-1 sums
    # grad_out[3+...]; check exact fp64 values against aten.
    grad = torch.tensor([[2.0**-60, 1.0, 2.0**60, 4.0, 8.0, 16.0]], dtype=torch.float64)
    x = torch.zeros(1, 3, dtype=torch.float64)
    out = replication_pad1d_backward(grad, x, (2, 1))
    ref = torch.ops.aten.replication_pad1d_backward(grad, x, (2, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out[0, 0] == 2.0**-60 + 1.0 + 2.0**60  # left edge, ascending order


def test_replication_pad1d_backward_tensor_padding():
    torch.manual_seed(0)
    x = torch.randn(2, 5, dtype=torch.float64)
    grad = torch.randn(2, 8, dtype=torch.float64)
    out = replication_pad1d_backward(grad, x, torch.tensor([1, 2]))
    ref = torch.ops.aten.replication_pad1d_backward(grad, x, (1, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_replication_pad1d_backward_validation_errors():
    x = torch.randn(2, 5)
    grad = torch.randn(2, 9)
    with pytest.raises(ValueError, match="expects 2D"):
        replication_pad1d_backward(torch.randn(6), torch.randn(6), (1, 1))
    with pytest.raises(
        RuntimeError, match="expected scalar type Double but found Float"
    ):
        replication_pad1d_backward(grad, x.double(), (2, 2))
    with pytest.raises(ValueError, match="grad_output has incorrect shape"):
        replication_pad1d_backward(torch.randn(2, 8), x, (2, 2))


def test_replication_pad1d_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.replication_pad1d_backward")
    launches = []
    kernel = mod.replication_pad1d_backward_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 9, dtype=torch.float64)
    grad = torch.randn(2, 3, 14, dtype=torch.float64)
    out = replication_pad1d_backward(grad, x, (2, 3))
    ref = torch.ops.aten.replication_pad1d_backward(grad, x, (2, 3))
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
