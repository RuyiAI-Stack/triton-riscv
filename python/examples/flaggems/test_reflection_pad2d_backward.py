import importlib

import pytest
import torch
import torch.nn.functional as F

from .reflection_pad2d_backward import reflection_pad2d_backward


@pytest.mark.parametrize(
    "shape, padding",
    [
        ((2, 5, 6), (1, 1, 1, 1)),
        ((1, 2, 5, 6), (2, 2, 1, 1)),
        ((2, 3, 6, 5), (3, 1, 2, 2)),
        ((1, 2, 4, 5), (2, 2, 2, 2)),  # corner cells collect all 9 terms
        ((2, 3, 4, 5), (0, 0, 0, 0)),  # zero padding
        ((2, 5, 6), (1, 2, 0, 3)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_reflection_pad2d_backward_matches_aten(shape, padding, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    oh = shape[-2] + padding[2] + padding[3]
    ow = shape[-1] + padding[0] + padding[1]
    grad = torch.randn(*shape[:-2], oh, ow, dtype=dtype)
    out = reflection_pad2d_backward(grad, x, padding)
    ref = torch.ops.aten.reflection_pad2d_backward(grad, x, padding)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad2d_backward_fp16_exact():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 5, dtype=torch.float16)
    grad = torch.randn(1, 2, 7, 7, dtype=torch.float16)
    out = reflection_pad2d_backward(grad, x, (1, 1, 2, 1))
    ref = torch.ops.aten.reflection_pad2d_backward(grad, x, (1, 1, 2, 1))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reflection_pad2d_backward_autograd_crosscheck():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 5, 6, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(2, 2, 10, 8, dtype=torch.float64)
    y = F.pad(x, (1, 1, 3, 2), mode="reflect")
    y.backward(grad)
    out = reflection_pad2d_backward(grad, x.detach(), (1, 1, 3, 2))
    torch.testing.assert_close(out, x.grad, rtol=0, atol=0)


def test_reflection_pad2d_backward_ordering_is_ascending_output():
    # Values chosen so different summation orders round differently in fp64;
    # only aten's ascending (tl, tc, tr, ml, mc, mr, bl, bc, br) order matches.
    torch.manual_seed(3)
    x = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    grad = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    out = reflection_pad2d_backward(grad, x, (2, 2, 2, 2))
    ref = torch.ops.aten.reflection_pad2d_backward(grad, x, (2, 2, 2, 2))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # corner input (1,1) receives 9 contributions
    expected = (
        grad[0, 0, 1, 1]
        + grad[0, 0, 1, 3]
        + grad[0, 0, 1, 7]
        + grad[0, 0, 3, 1]
        + grad[0, 0, 3, 3]
        + grad[0, 0, 3, 7]
        + grad[0, 0, 7, 1]
        + grad[0, 0, 7, 3]
        + grad[0, 0, 7, 7]
    )
    assert out[0, 0, 1, 1] == expected


def test_reflection_pad2d_backward_validation_errors():
    x = torch.randn(1, 2, 4, 5)
    with pytest.raises(ValueError, match="sequence of 4 elements"):
        reflection_pad2d_backward(torch.randn(1, 2, 6, 7), x, (1, 1, 1))


def test_reflection_pad2d_backward_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.reflection_pad2d_backward")
    launches = []
    kernel = mod.reflection_pad2d_backward_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(2, 2, 6, 7, dtype=torch.float64)
    grad = torch.randn(2, 2, 9, 9, dtype=torch.float64)
    out = reflection_pad2d_backward(grad, x, (1, 1, 2, 1))
    ref = torch.ops.aten.reflection_pad2d_backward(grad, x, (1, 1, 2, 1))
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
