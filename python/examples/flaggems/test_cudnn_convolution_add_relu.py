import importlib
import inspect

import pytest
import torch

from .cudnn_convolution_add_relu import cudnn_convolution_add_relu


def _oracle(x, w, z, alpha=1.0, bias=None, stride=1, padding=0, dilation=1, groups=1):
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
        conv = conv + bias.view((1, -1) + (1,) * spatial)
    return torch.relu(conv + alpha * z)


def test_cudnn_convolution_add_relu_2d():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 7)
    w = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    z = torch.randn(2, 4, 5, 5)
    ref = _oracle(x, w, z, bias=bias)
    tri = cudnn_convolution_add_relu(x, w, z, bias=bias)
    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_cudnn_convolution_add_relu_alpha_and_defaults():
    torch.manual_seed(0)
    x = torch.randn(2, 2, 6, 6)
    w = torch.randn(2, 2, 3, 3)
    z = torch.randn(2, 2, 4, 4)
    ref = _oracle(x, w, z, alpha=2.5)
    tri = cudnn_convolution_add_relu(x, w, z, alpha=2.5)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_cudnn_convolution_add_relu_bad_spatial_dims():
    with pytest.raises(ValueError):
        cudnn_convolution_add_relu(
            torch.randn(2, 3), torch.randn(1, 3), torch.randn(2, 1)
        )


def test_cudnn_convolution_add_relu_docstring_declares_delegation():
    doc = inspect.getdoc(cudnn_convolution_add_relu) or ""
    assert "delegated to aten" in doc
    assert "epilogue" in doc


def test_cudnn_convolution_add_relu_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.cudnn_convolution_add_relu")
    launches = []
    real_run = mod._cudnn_convolution_add_relu_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._cudnn_convolution_add_relu_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 7)
    w = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    z = torch.randn(2, 4, 5, 5)
    out = mod.cudnn_convolution_add_relu(x, w, z, bias=bias)
    assert len(launches) >= 1, "the epilogue must launch the Triton kernel"
    torch.testing.assert_close(out, _oracle(x, w, z, bias=bias), rtol=0.0, atol=1e-5)


def test_cudnn_convolution_add_relu_rejects_fp64():
    # Mirrors the upstream dtype gate (RuntimeError flavor).
    with pytest.raises(RuntimeError):
        cudnn_convolution_add_relu(
            torch.randn(1, 1, 5, 5, dtype=torch.float64),
            torch.randn(1, 1, 3, 3, dtype=torch.float64),
            torch.randn(1, 1, 3, 3, dtype=torch.float64),
        )
