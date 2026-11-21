import importlib

import pytest
import torch

from ._addmm_activation import _addmm_activation


def _bias_mat1_mat2():
    torch.manual_seed(0)
    bias = torch.randn(3)
    mat1 = torch.randn(2, 4)
    mat2 = torch.randn(4, 3)
    return bias, mat1, mat2


def _tolerance(dtype):
    # fp64 keeps a native double accumulator; fp16 runs the fp32 compute
    # path (suite band); fp32 accumulates in fp32 with a different order
    # than aten, so 1e-6 on these small K.
    if dtype == torch.float16:
        return 1e-2, 1e-3
    if dtype == torch.float64:
        return 0, 1e-9
    return 0, 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_addmm_activation_relu_matches_aten(dtype):
    torch.manual_seed(0)
    bias = torch.randn(3, dtype=dtype)
    mat1 = torch.randn(2, 4, dtype=dtype)
    mat2 = torch.randn(4, 3, dtype=dtype)
    ref = torch.ops.aten._addmm_activation(bias, mat1, mat2, use_gelu=False)
    tri = _addmm_activation(bias, mat1, mat2)
    assert tri.dtype == ref.dtype and tri.shape == ref.shape
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_addmm_activation_gelu_matches_aten():
    bias, mat1, mat2 = _bias_mat1_mat2()
    ref = torch.ops.aten._addmm_activation(bias, mat1, mat2, use_gelu=True)
    tri = _addmm_activation(bias, mat1, mat2, use_gelu=True)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-6)


def test_addmm_activation_beta_alpha():
    # Probed on torch CPU: the coefficient layout is
    # beta * bias + alpha * (mat1 @ mat2), then the activation.
    bias, mat1, mat2 = _bias_mat1_mat2()
    ref = torch.relu(2.0 * bias + 3.0 * (mat1 @ mat2))
    tri = _addmm_activation(bias, mat1, mat2, beta=2.0, alpha=3.0)
    torch.testing.assert_close(tri, ref, rtol=0.0, atol=1e-5)


def test_addmm_activation_out_buffer():
    bias, mat1, mat2 = _bias_mat1_mat2()
    out = torch.empty(2, 3)
    ret = _addmm_activation(bias, mat1, mat2, out=out)
    assert ret is out
    ref = torch.ops.aten._addmm_activation(bias, mat1, mat2, use_gelu=False)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)


def test_addmm_activation_bad_shapes():
    bias = torch.randn(3)
    with pytest.raises((RuntimeError, AssertionError)):
        _addmm_activation(bias, torch.randn(2, 3), torch.randn(4, 3))


def test_addmm_activation_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._addmm_activation")
    launches = []
    real_run = mod._addmm_activation_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._addmm_activation_kernel, "run", spy_run)
    bias, mat1, mat2 = _bias_mat1_mat2()
    out = mod._addmm_activation(bias, mat1, mat2)
    assert len(launches) >= 1, "_addmm_activation must launch the Triton kernel"
    ref = torch.ops.aten._addmm_activation(bias, mat1, mat2, use_gelu=False)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)
