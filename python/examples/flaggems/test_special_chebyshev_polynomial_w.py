import importlib

import pytest
import torch

from .special_chebyshev_polynomial_w import special_chebyshev_polynomial_w


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_chebyshev_polynomial_w_scalar_n(n):
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 1.5

    ref = torch.special.chebyshev_polynomial_w(x, n)
    tri = special_chebyshev_polynomial_w(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: the fourth-kind recurrence W_k = 2x W_{k-1} -
    # W_{k-2}, W_0 = 1, W_1 = 2x + 1, on fp32; 1e-5 relative up to degree 10.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_chebyshev_polynomial_w_tensor_n():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32) * 1.5
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.chebyshev_polynomial_w(x.unsqueeze(1), n)
    tri = special_chebyshev_polynomial_w(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_chebyshev_polynomial_w_known_rows():
    # W_0 = 1, W_1 = 2x + 1, W_2 = 4x^2 + 2x - 1 (probed on torch CPU
    # fp32: W_2(0.5) = 1, W_2(-0.5) = -1), n < 0 = 0.
    x = torch.tensor([0.5, -0.5], dtype=torch.float32)

    assert special_chebyshev_polynomial_w(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        special_chebyshev_polynomial_w(x, 1), 2.0 * x + 1.0, rtol=0, atol=0
    )
    torch.testing.assert_close(
        special_chebyshev_polynomial_w(x, 2),
        torch.tensor([1.0, -1.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_chebyshev_polynomial_w(x, -1), torch.zeros_like(x), rtol=0, atol=0
    )


def test_special_chebyshev_polynomial_w_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_chebyshev_polynomial_w")
    launches = []
    real_run = mod._w_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._w_flat_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    out = mod.special_chebyshev_polynomial_w(x, 4)
    assert len(launches) >= 1, "special_chebyshev_polynomial_w must launch a kernel"
    torch.testing.assert_close(
        out, torch.special.chebyshev_polynomial_w(x, 4), rtol=1e-5, atol=1e-5
    )
