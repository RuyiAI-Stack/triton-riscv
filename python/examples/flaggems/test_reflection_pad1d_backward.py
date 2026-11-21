import importlib

import pytest
import torch
import torch.nn.functional as F

from .reflection_pad1d_backward import reflection_pad1d_backward


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((2, 5), (2, 2)),
        ((3, 6), (1, 2)),
        ((2, 3, 7), (3, 1)),
        ((2, 4), (0, 0)),  # zero padding: copy path
        ((1, 4), (3, 2)),  # near-full wrap, asymmetric
        ((2, 5), (2, 2)),  # input w=2 receives left+center+right
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_reflection_pad1d_backward_matches_aten(shape, padding, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    grad = torch.randn(*shape[:-1], shape[-1] + padding[0] + padding[1], dtype=dtype)
    out = reflection_pad1d_backward(grad, x, padding)
    ref = torch.ops.aten.reflection_pad1d_backward(grad, x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad1d_backward_fp16_exact():
    torch.manual_seed(0)
    x = torch.randn(2, 5, dtype=torch.float16)
    grad = torch.randn(2, 9, dtype=torch.float16)
    out = reflection_pad1d_backward(grad, x, (2, 2))
    ref = torch.ops.aten.reflection_pad1d_backward(grad, x, (2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad1d_backward_autograd_crosscheck():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(2, 3, 10, dtype=torch.float64)
    y = F.pad(x, (2, 2), mode="reflect")
    y.backward(grad)
    out = reflection_pad1d_backward(grad, x.detach(), (2, 2))
    torch.testing.assert_close(out, x.grad, rtol=0, atol=0)


def test_reflection_pad1d_backward_multi_contribution_exact():
    # w=2 with pads (2,2) on W=4: center+left+right land on one element.
    grad = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]], dtype=torch.float64
    )
    x = torch.zeros(1, 4, dtype=torch.float64)
    out = reflection_pad1d_backward(grad, x, (2, 2))
    ref = torch.ops.aten.reflection_pad1d_backward(grad, x, (2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out[0, 2] == 1.0 + 16.0 + 64.0  # left@0 + center@4 + right@6


def test_reflection_pad1d_backward_empty_batch():
    x = torch.empty(0, 5, dtype=torch.float64)
    grad = torch.empty(0, 9, dtype=torch.float64)
    out = reflection_pad1d_backward(grad, x, (2, 2))
    assert out.shape == (0, 5)


def test_reflection_pad1d_backward_validation_errors():
    x = torch.randn(2, 5)
    grad = torch.randn(2, 9)
    with pytest.raises(ValueError, match="sequence of length 2"):
        reflection_pad1d_backward(grad, x, (1, 1, 1))
    with pytest.raises(ValueError, match="must be >= 0"):
        reflection_pad1d_backward(grad, x, (-1, 2))
    with pytest.raises(ValueError, match="at least 1 dimension"):
        reflection_pad1d_backward(torch.tensor(1.0), torch.tensor(1.0), (1, 1))
    with pytest.raises(ValueError, match="at least 2"):
        reflection_pad1d_backward(torch.randn(2, 3), torch.randn(2, 1), (1, 1))
    with pytest.raises(ValueError, match="less than the input width"):
        reflection_pad1d_backward(torch.randn(2, 7), torch.randn(2, 5), (5, 0))


def test_reflection_pad1d_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.reflection_pad1d_backward")
    launches = []
    kernel = mod.reflection_pad1d_backward_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 9, dtype=torch.float64)
    grad = torch.randn(2, 3, 14, dtype=torch.float64)
    out = reflection_pad1d_backward(grad, x, (2, 3))
    ref = torch.ops.aten.reflection_pad1d_backward(grad, x, (2, 3))
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
