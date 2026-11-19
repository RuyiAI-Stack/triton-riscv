import importlib

import pytest
import torch

from .special_laguerre_polynomial_l import special_laguerre_polynomial_l


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_laguerre_polynomial_l_scalar_n(n):
    torch.manual_seed(0)
    x = torch.rand(512, dtype=torch.float32) * 10.0

    ref = torch.special.laguerre_polynomial_l(x, n)
    tri = special_laguerre_polynomial_l(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: Laguerre recurrence L_k = ((2k-1-x) L_{k-1} -
    # (k-1) L_{k-2}) / k in fp32; 1e-5 relative up to degree 10.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_laguerre_polynomial_l_tensor_n():
    torch.manual_seed(0)
    x = torch.rand(64, dtype=torch.float32) * 5.0
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.laguerre_polynomial_l(x.unsqueeze(1), n)
    tri = special_laguerre_polynomial_l(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_laguerre_polynomial_l_known_rows():
    # L_0 = 1, L_1 = 1 - x, L_2 = (x^2 - 4x + 2)/2 (probed on torch CPU
    # fp32: L_2(0.5) = 0.125, L_2(3) = -0.5), n < 0 = 0.
    x = torch.tensor([0.5, 3.0], dtype=torch.float32)

    assert special_laguerre_polynomial_l(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        special_laguerre_polynomial_l(x, 1), 1.0 - x, rtol=0, atol=0
    )
    torch.testing.assert_close(
        special_laguerre_polynomial_l(x, 2),
        torch.tensor([0.125, -0.5]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_laguerre_polynomial_l(x, -1), torch.zeros_like(x), rtol=0, atol=0
    )


def test_special_laguerre_polynomial_l_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_laguerre_polynomial_l")
    launches = []
    real_run = mod._laguerre_flat_tensor_tensor_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._laguerre_flat_tensor_tensor_kernel, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 5.0
    n = torch.full((1024,), 3, dtype=torch.int64)
    out = mod.special_laguerre_polynomial_l(x, n)
    assert len(launches) >= 1, "special_laguerre_polynomial_l must launch a kernel"
    torch.testing.assert_close(
        out, torch.special.laguerre_polynomial_l(x, n), rtol=1e-5, atol=1e-5
    )
