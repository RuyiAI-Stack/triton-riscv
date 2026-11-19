import importlib

import pytest
import torch

from .special_hermite_polynomial_h import special_hermite_polynomial_h


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_special_hermite_polynomial_h_scalar_n(n):
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32) * 2.0

    ref = torch.special.hermite_polynomial_h(x, n)
    tri = special_hermite_polynomial_h(x, n)

    assert tri.dtype == ref.dtype
    # Tolerance source: the reference itself evaluates the recurrence in
    # fp32 (probed: switching our kernel to fp64 drifts 3.4e-5 away from
    # torch at n = 10), so the port keeps fp32 and inherits the same
    # trajectory; measured worst relative error is 1.9e-6 for scalar
    # n <= 10 and 1.3e-5 for tensor n up to 15 -- 5e-5 keeps > 3x margin.
    torch.testing.assert_close(tri, ref, rtol=5e-5, atol=5e-5)


def test_special_hermite_polynomial_h_tensor_n():
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32) * 2.0
    n = torch.arange(0, 16, dtype=torch.int64)

    ref = torch.special.hermite_polynomial_h(x.unsqueeze(1), n)
    tri = special_hermite_polynomial_h(x.unsqueeze(1), n)

    assert tri.shape == ref.shape
    torch.testing.assert_close(tri, ref, rtol=5e-5, atol=5e-5)


def test_special_hermite_polynomial_h_known_rows():
    # H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2 (probed on torch CPU fp32),
    # n < 0 = 0.
    x = torch.tensor([0.5, 2.0], dtype=torch.float32)

    assert special_hermite_polynomial_h(x, 0).tolist() == [1.0, 1.0]
    torch.testing.assert_close(
        special_hermite_polynomial_h(x, 1), 2.0 * x, rtol=0, atol=0
    )
    torch.testing.assert_close(
        special_hermite_polynomial_h(x, 2),
        torch.tensor([-1.0, 14.0]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        special_hermite_polynomial_h(x, -1), torch.zeros_like(x), rtol=0, atol=0
    )


def test_special_hermite_polynomial_h_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_hermite_polynomial_h")
    launches = []
    real_run = mod._hermite_tensor_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._hermite_tensor_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32)
    out = mod.special_hermite_polynomial_h(x, 4)
    assert len(launches) >= 1, "special_hermite_polynomial_h must launch a kernel"
    torch.testing.assert_close(
        out, torch.special.hermite_polynomial_h(x, 4), rtol=1e-5, atol=1e-5
    )
