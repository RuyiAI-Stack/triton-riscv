import importlib

import pytest
import torch

from .special_shifted_chebyshev_polynomial_t import (
    special_shifted_chebyshev_polynomial_t,
)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_shifted_chebyshev_polynomial_t_scalar_n(n):
    torch.manual_seed(0)
    x = torch.rand(512, dtype=torch.float32) * 2.0 - 0.5

    ref = torch.special.shifted_chebyshev_polynomial_t(x, n)
    tri = special_shifted_chebyshev_polynomial_t(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: the same recurrence with the shifted argument
    # (2x - 1) in fp32; 1e-5 relative up to degree 10.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_shifted_chebyshev_polynomial_t_tensor_n():
    torch.manual_seed(0)
    x = torch.rand(64, dtype=torch.float32) * 2.0 - 0.5
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.shifted_chebyshev_polynomial_t(x.unsqueeze(1), n)
    tri = special_shifted_chebyshev_polynomial_t(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_shifted_chebyshev_polynomial_t_known_rows():
    # T*_0 = 1, T*_1 = 2x - 1, T*_2 = 8x^2 - 8x + 1 (probed on torch CPU
    # fp32: T*_2(0.5) = -1, T*_2(-0.5) = 7), n < 0 = 0.
    x = torch.tensor([0.5, -0.5], dtype=torch.float32)

    assert special_shifted_chebyshev_polynomial_t(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        special_shifted_chebyshev_polynomial_t(x, 1), 2.0 * x - 1.0, rtol=0, atol=0
    )
    torch.testing.assert_close(
        special_shifted_chebyshev_polynomial_t(x, 2),
        torch.tensor([-1.0, 7.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_shifted_chebyshev_polynomial_t(x, -1),
        torch.zeros_like(x),
        rtol=0,
        atol=0,
    )


def test_special_shifted_chebyshev_polynomial_t_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_shifted_chebyshev_polynomial_t")
    launches = []
    real_run = mod.shifted_chebyshev_polynomial_t_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.shifted_chebyshev_polynomial_t_kernel, "run", spy_run)
    x = torch.rand(1024, dtype=torch.float32) * 2.0 - 0.5
    out = mod.special_shifted_chebyshev_polynomial_t(x, 4)
    assert len(launches) >= 1, "shifted chebyshev_t must launch a kernel"
    torch.testing.assert_close(
        out,
        torch.special.shifted_chebyshev_polynomial_t(x, 4),
        rtol=1e-5,
        atol=1e-5,
    )
