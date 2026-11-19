import importlib

import pytest
import torch

from .special_chebyshev_polynomial_t import special_chebyshev_polynomial_t


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_chebyshev_polynomial_t_scalar_n(n):
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 1.5

    ref = torch.special.chebyshev_polynomial_t(x, n)
    tri = special_chebyshev_polynomial_t(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: the forward recurrence T_k = 2x T_{k-1} - T_{k-2}
    # evaluated in fp32; 1e-5 relative covers degrees up to 10 including
    # |x| > 1, where the polynomial grows geometrically.
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_chebyshev_polynomial_t_tensor_n():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32) * 1.5
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.chebyshev_polynomial_t(x.unsqueeze(1), n)
    tri = special_chebyshev_polynomial_t(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-5)


def test_special_chebyshev_polynomial_t_known_rows():
    # T_0 = 1, T_1 = x, T_2 = 2x^2 - 1 (probed on torch CPU fp32), and
    # n < 0 degrades to 0 for every family in the special namespace.
    x = torch.tensor([0.5, 2.0], dtype=torch.float32)

    assert special_chebyshev_polynomial_t(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(special_chebyshev_polynomial_t(x, 1), x, rtol=0, atol=0)
    torch.testing.assert_close(
        special_chebyshev_polynomial_t(x, 2),
        torch.tensor([-0.5, 7.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_chebyshev_polynomial_t(x, -1), torch.zeros_like(x), rtol=0, atol=0
    )


def test_special_chebyshev_polynomial_t_non_contiguous_and_empty():
    base = torch.randn(2048, dtype=torch.float32)
    x = base[::2]

    assert not x.is_contiguous()
    torch.testing.assert_close(
        special_chebyshev_polynomial_t(x, 3),
        torch.special.chebyshev_polynomial_t(x.clone(), 3),
        rtol=1e-5,
        atol=1e-5,
    )
    assert special_chebyshev_polynomial_t(torch.tensor([]), 3).numel() == 0


def test_special_chebyshev_polynomial_t_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_chebyshev_polynomial_t")
    launches = []
    real_run = mod._cheb_t_small_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._cheb_t_small_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    out = mod.special_chebyshev_polynomial_t(x, 4)
    assert len(launches) >= 1, "special_chebyshev_polynomial_t must launch a kernel"
    torch.testing.assert_close(
        out, torch.special.chebyshev_polynomial_t(x, 4), rtol=1e-5, atol=1e-5
    )
