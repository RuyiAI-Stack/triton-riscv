import importlib

import pytest
import torch

from .diagflat import diagflat


def test_diagflat_1d_matches_aten():
    x = torch.arange(4, dtype=torch.float64)
    out = diagflat(x)
    torch.testing.assert_close(out, torch.diagflat(x), rtol=0, atol=0)
    assert out.shape == (4, 4)


def test_diagflat_offsets_match_aten():
    x = torch.arange(3, dtype=torch.float64)
    for k in (1, -1, 3, -4):
        out = diagflat(x, k)
        ref = torch.diagflat(x, k)
        assert out.shape == ref.shape  # (n+|k|)^2, non-square padding cases
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"k={k}")


def test_diagflat_2d_input_flattens():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
    out = diagflat(x)
    torch.testing.assert_close(out, torch.diagflat(x), rtol=0, atol=0)


def test_diagflat_empty_input():
    out = diagflat(torch.tensor([]))
    ref = torch.diagflat(torch.tensor([]))
    assert out.shape == ref.shape == (0, 0)
    assert out.dtype == ref.dtype


def test_diagflat_strided_1d_input():
    s = torch.arange(10, dtype=torch.float64)[::2]
    assert s.stride(0) == 2
    out = diagflat(s, 1)
    torch.testing.assert_close(out, torch.diagflat(s, 1), rtol=0, atol=0)


def test_diagflat_dtypes():
    for x in (
        torch.arange(3, dtype=torch.int64),
        torch.arange(3, dtype=torch.float32),
        torch.tensor([True, False]),
    ):
        out = diagflat(x, -1)
        ref = torch.diagflat(x, -1)
        assert out.dtype == ref.dtype
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_diagflat_complex_input():
    x = torch.complex(torch.arange(3.0), -torch.arange(3.0))
    out = diagflat(x, 1)
    torch.testing.assert_close(out, torch.diagflat(x, 1), rtol=0, atol=0)
    nc = torch.complex(torch.arange(6.0).reshape(2, 3).t(), torch.ones(2, 3).t())
    out2 = diagflat(nc)
    torch.testing.assert_close(out2, torch.diagflat(nc), rtol=0, atol=0)


def test_diagflat_errors_match_aten():
    with pytest.raises(
        TypeError, match="argument 'offset' \\(position 2\\) must be int, not float"
    ):
        diagflat(torch.arange(3), 1.5)
    with pytest.raises(
        TypeError, match="argument 'input' \\(position 1\\) must be Tensor, not str"
    ):
        diagflat("x")


def test_diagflat_gradient_matches_aten():
    x = torch.arange(4, dtype=torch.float64, requires_grad=True)
    g = torch.arange(25, dtype=torch.float64).reshape(5, 5)
    out = diagflat(x, 1)
    out.backward(g)
    xr = torch.arange(4, dtype=torch.float64, requires_grad=True)
    torch.diagflat(xr, 1).backward(g)
    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)


def _spy_on(kernel, launches):
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    return spy_run


def test_diagflat_triton_path_fill_and_scatter(monkeypatch):
    mod = importlib.import_module("flaggems.diagflat")
    x = torch.arange(4, dtype=torch.float64)
    for kname in ("_diagflat_zero_kernel", "_diagflat_scatter_kernel"):
        launches = []
        kernel = getattr(mod, kname)
        monkeypatch.setattr(kernel, "run", _spy_on(kernel, launches))
        out = diagflat(x, 1)
        assert len(launches) >= 1, f"{kname} must be launched"
        torch.testing.assert_close(out, torch.diagflat(x, 1), rtol=0, atol=0)


def test_diagflat_triton_path_backward_gather(monkeypatch):
    mod = importlib.import_module("flaggems.diagflat")
    launches = []
    kernel = mod._diagflat_gather_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.arange(4, dtype=torch.float64, requires_grad=True)
    out = diagflat(x, 1)
    out.backward(torch.ones_like(out))
    assert len(launches) >= 1
    expected = torch.zeros(4, dtype=torch.float64)
    expected[:4] = 1.0
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
