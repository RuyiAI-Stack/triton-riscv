import importlib

import pytest
import torch

from .special_xlogy import special_xlogy, special_xlogy_


def _tolerance(dtype):
    # Tolerance source: same construction and bounds as the xlogy port
    # (kernel evaluates x * log(y) in fp32 for every input dtype) --
    # fp32 1e-4 (test_arctan2.py convention), fp16 1e-3, fp64 1e-6.
    if dtype == torch.float16:
        return 1e-3, 1e-3
    if dtype == torch.float64:
        return 1e-6, 1e-6
    return 1e-4, 1e-4


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_xlogy(dtype, size):
    torch.manual_seed(0)
    # Random y includes negatives: both sides produce NaN there.
    x = torch.randn(size, dtype=dtype) * 10.0
    y = torch.randn(size, dtype=dtype) * 5.0

    ref = torch.special.xlogy(x, y)
    tri = special_xlogy(x, y)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize("shape_x, shape_y", [((16, 256), (256,)), ((4, 1), (4, 128))])
def test_special_xlogy_broadcast(shape_x, shape_y):
    torch.manual_seed(0)
    x = torch.randn(shape_x, dtype=torch.float32) * 10.0
    y = torch.randn(shape_y, dtype=torch.float32) * 5.0

    ref = torch.special.xlogy(x, y)
    tri = special_xlogy(x, y)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_special_xlogy_inplace():
    x = torch.randn(256, dtype=torch.float32) * 10.0
    y = torch.rand(256, dtype=torch.float32) * 5.0 + 0.1

    ref = torch.special.xlogy(x, y)
    tri = x.clone()
    out = special_xlogy_(tri, y)

    assert out is tri
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_special_xlogy_non_contiguous():
    base_x = torch.randn(2048, dtype=torch.float32) * 10.0
    base_y = torch.rand(2048, dtype=torch.float32) * 5.0 + 0.1
    x, y = base_x[::2], base_y[::2]

    assert not x.is_contiguous() and not y.is_contiguous()
    torch.testing.assert_close(
        special_xlogy(x, y),
        torch.special.xlogy(x.clone(), y.clone()),
        rtol=1e-4,
        atol=1e-4,
    )


def test_special_xlogy_special_values():
    # ATen precedence probed on torch CPU fp32 (same table as the xlogy
    # port): y is NaN -> NaN even for x == 0; x == ±0 -> ±0 for every
    # other y including -inf; otherwise plain IEEE x * log(y).
    pos0, neg0 = 0.0, -0.0
    inf, ninf = float("inf"), float("-inf")
    nan = float("nan")
    x = torch.tensor([pos0, pos0, pos0, neg0, 1.0, 1.0, -1.0, nan, inf])
    y = torch.tensor([nan, ninf, 2.0, 2.0, 0.0, ninf, 2.0, 2.0, 2.0])

    ref = torch.special.xlogy(x, y)
    tri = special_xlogy(x, y)

    # Every row lands on an exact value (0 / ±inf / NaN).
    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)


def test_special_xlogy_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_xlogy")
    launches = []
    real_run = mod.special_xlogy_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.special_xlogy_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 10.0
    y = torch.rand(1024, dtype=torch.float32) * 5.0 + 0.1
    out = mod.special_xlogy(x, y)
    assert len(launches) >= 1, "special_xlogy must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.xlogy(x, y), rtol=1e-4, atol=1e-4)
