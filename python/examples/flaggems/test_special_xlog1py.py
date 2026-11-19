import importlib

import pytest
import torch

from .special_xlog1py import special_xlog1py


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_xlog1py(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype) * 10.0
    y = torch.rand(size, dtype=dtype) * 5.0 - 0.5

    ref = torch.special.xlog1py(x, y)
    tri = special_xlog1py(x, y)

    assert tri.dtype == ref.dtype
    # Tolerance source: xlog1py composes x * log1p(y) in fp32, so it
    # inherits the xlogy port bounds: fp32 1e-4, fp16 1e-3, fp64 1e-6.
    if dtype == torch.float16:
        torch.testing.assert_close(tri, ref, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif dtype == torch.float64:
        torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-6, equal_nan=True)
    else:
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("shape_x, shape_y", [((16, 256), (256,)), ((4, 1), (4, 128))])
def test_special_xlog1py_broadcast(shape_x, shape_y):
    torch.manual_seed(0)
    x = torch.randn(shape_x, dtype=torch.float32) * 10.0
    y = torch.rand(shape_y, dtype=torch.float32) * 5.0 - 0.5

    ref = torch.special.xlog1py(x, y)
    tri = special_xlog1py(x, y)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_special_xlog1py_special_values():
    # ATen contract probed on torch CPU fp32: xlog1py(0, y) = 0 for every
    # y <= -1 (the 0 * log term is defined away), y < -1 = nan for x != 0,
    # and the y = NaN row stays NaN.
    pos0, neg0 = 0.0, -0.0
    inf = float("inf")
    nan = float("nan")
    x = torch.tensor([pos0, pos0, neg0, 1.0, 1.0, -1.0, 1.0, nan])
    y = torch.tensor([-1.0, -2.0, -1.0, -1.0, -2.0, 0.5, inf, 2.0])

    ref = torch.special.xlog1py(x, y)
    tri = special_xlog1py(x, y)

    # The sentinel rows (0 / -inf / nan) are compared exactly; the two
    # finite rows carry the log1p ulp difference, so they get tolerance.
    torch.testing.assert_close(tri, ref, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert tri[0].item() == 0.0 and tri[1].item() == 0.0 and tri[2].item() == 0.0
    assert tri[3].item() == float("-inf") and tri[4].item() != tri[4].item()


def test_special_xlog1py_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_xlog1py")
    launches = []
    real_run = mod.xlog1py_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.xlog1py_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 10.0
    y = torch.rand(1024, dtype=torch.float32) * 5.0 - 0.5
    out = mod.special_xlog1py(x, y)
    assert len(launches) >= 1, "special_xlog1py must launch the Triton kernel"
    torch.testing.assert_close(
        out, torch.special.xlog1py(x, y), rtol=1e-4, atol=1e-4, equal_nan=True
    )
