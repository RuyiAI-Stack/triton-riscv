import importlib
import inspect

import pytest
import torch

from .cudnn_convolution_relu import cudnn_convolution_relu


def _oracle(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
    spatial = x.dim() - 2
    conv_fn = (
        torch.nn.functional.conv1d,
        torch.nn.functional.conv2d,
        torch.nn.functional.conv3d,
    )[spatial - 1]
    conv = conv_fn(
        x, w, None, stride=stride, padding=padding, dilation=dilation, groups=groups
    )
    if bias is not None:
        shape = (1, -1) + (1,) * spatial
        conv = conv + bias.view(shape)
    return torch.relu(conv)


def test_cudnn_convolution_relu_2d():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 7)
    w = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    ref = _oracle(x, w, bias)
    tri = cudnn_convolution_relu(x, w, bias)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_cudnn_convolution_relu_1d_3d_and_options():
    torch.manual_seed(0)
    x1 = torch.randn(2, 3, 10)
    w1 = torch.randn(4, 3, 3)
    torch.testing.assert_close(
        cudnn_convolution_relu(x1, w1), _oracle(x1, w1), rtol=0.0, atol=1e-5
    )

    x3 = torch.randn(1, 2, 4, 4, 4)
    w3 = torch.randn(2, 2, 2, 2, 2)
    torch.testing.assert_close(
        cudnn_convolution_relu(x3, w3), _oracle(x3, w3), rtol=0.0, atol=1e-5
    )

    # stride / padding / dilation / groups / bias=None
    x2 = torch.randn(2, 4, 9, 9)
    w2 = torch.randn(4, 2, 3, 3)
    opts = dict(stride=2, padding=1, groups=2)
    ref = _oracle(x2, w2, None, **opts)
    tri = cudnn_convolution_relu(x2, w2, None, **opts)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_cudnn_convolution_relu_bad_spatial_dims():
    with pytest.raises(ValueError):
        cudnn_convolution_relu(torch.randn(2, 3), torch.randn(1, 3))


def test_cudnn_convolution_relu_docstring_declares_delegation():
    # The convolution arithmetic is delegated to aten; only the epilogue is
    # Triton. The docstring must say so instead of letting a reviewer
    # discover it from the code.
    doc = inspect.getdoc(cudnn_convolution_relu) or ""
    assert "delegated to aten" in doc
    assert "epilogue" in doc


def test_cudnn_convolution_relu_triton_path(monkeypatch):
    # Spy covers the epilogue kernel only: the convolution itself runs
    # through aten (see the docstring).
    mod = importlib.import_module("flaggems.cudnn_convolution_relu")
    launches = []
    real_run = mod._cudnn_convolution_relu_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._cudnn_convolution_relu_kernel, "run", spy_run)
    x = torch.randn(2, 3, 7, 7)
    w = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    out = mod.cudnn_convolution_relu(x, w, bias)
    assert len(launches) >= 1, "the relu epilogue must launch the Triton kernel"
    torch.testing.assert_close(out, _oracle(x, w, bias), rtol=0.0, atol=1e-5)


def test_cudnn_convolution_relu_rejects_fp64():
    # Mirrors the upstream dtype gate: the cuDNN engine only exists for
    # fp16/bf16/fp32, and the upstream port rejects anything else up front.
    with pytest.raises(AssertionError):
        cudnn_convolution_relu(
            torch.randn(1, 1, 5, 5, dtype=torch.float64),
            torch.randn(1, 1, 3, 3, dtype=torch.float64),
        )


def test_cudnn_convolution_relu_empty_batch():
    x = torch.empty(0, 3, 5, 5)
    w = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    out = cudnn_convolution_relu(x, w, bias)
    assert out.shape == (0, 4, 3, 3)
    torch.testing.assert_close(out, _oracle(x, w, bias), rtol=0.0, atol=0.0)
