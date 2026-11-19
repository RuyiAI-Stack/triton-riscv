import importlib

import pytest
import torch

from .special_hermite_polynomial_he import special_hermite_polynomial_he


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_hermite_polynomial_he_scalar_n(n):
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 2.0

    ref = torch.special.hermite_polynomial_he(x, n)
    tri = special_hermite_polynomial_he(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: same fp32-recurrence contract as the physicist
    # Hermite test (the reference evaluates in fp32 too); measured worst
    # 1.9e-6 up to degree 10, so 5e-5 keeps > 3x margin for the tensor
    # degrees beyond it.
    torch.testing.assert_close(tri, ref, rtol=5e-5, atol=5e-5)


def test_special_hermite_polynomial_he_tensor_n():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32) * 2.0
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.hermite_polynomial_he(x.unsqueeze(1), n)
    tri = special_hermite_polynomial_he(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=5e-5, atol=5e-5)


def test_special_hermite_polynomial_he_known_rows():
    # He_0 = 1, He_1 = x, He_2 = x^2 - 1 (probed on torch CPU fp32),
    # n < 0 = 0.
    x = torch.tensor([0.5, 2.0], dtype=torch.float32)

    assert special_hermite_polynomial_he(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(special_hermite_polynomial_he(x, 1), x, rtol=0, atol=0)
    torch.testing.assert_close(
        special_hermite_polynomial_he(x, 2),
        torch.tensor([-0.75, 3.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_hermite_polynomial_he(x, -1), torch.zeros_like(x), rtol=0, atol=0
    )


def test_special_hermite_polynomial_he_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_hermite_polynomial_he")
    launches = []
    real_run = mod.hermite_he_func.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.hermite_he_func, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    out = mod.special_hermite_polynomial_he(x, 4)
    assert len(launches) >= 1, "special_hermite_polynomial_he must launch a kernel"
    torch.testing.assert_close(
        out, torch.special.hermite_polynomial_he(x, 4), rtol=1e-5, atol=1e-5
    )
