import importlib

import pytest
import torch

from .special_i1e import special_i1e


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_special_i1e(size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=torch.float32) * 8.0

    ref = torch.special.i1e(x)
    tri = special_i1e(x)

    assert tri.dtype == ref.dtype
    # Tolerance source, measured over 4096 samples per band: the minimax
    # polynomial only pins i1e to absolute accuracy near its zero, so the
    # sub-unit band reaches 2.1e-4 relative on tiny values (the 2e-6 atol
    # covers those), and the polynomial/series seam at |x| = 8 departs by
    # 1.4e-5 relative; 5e-5 keeps a 3.6x margin on the seam.
    torch.testing.assert_close(tri, ref, rtol=5e-5, atol=2e-6)


def test_special_i1e_branches_and_decay():
    # The scaling keeps i1e bounded: it rises to ~0.5 near x = 1.5 and
    # decays toward 0 for large x instead of overflowing like i1.
    x = torch.cat(
        [
            torch.linspace(-20.0, 20.0, 401),
            torch.tensor([3.75, 3.7500001, 20.0, 50.0, 88.0]),
        ]
    )

    ref = torch.special.i1e(x)
    tri = special_i1e(x)

    torch.testing.assert_close(tri, ref, rtol=1e-5, atol=1e-6)
    assert tri[-1].abs().item() < 0.05


def test_special_i1e_special_values():
    # ATen contract probed on torch CPU fp32: i1e(±0) = ±0, i1e(+inf) = 0,
    # i1e(-inf) = -0, nan stays nan.
    x = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
    )

    ref = torch.special.i1e(x)
    tri = special_i1e(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0 and tri[2].item() == 0.0
    assert tri[3].item() == 0.0 and tri[3].signbit()


def test_special_i1e_weak_bands():
    # The two bands the tolerance comment quantifies: the zero at 0 (the
    # polynomial's absolute-accuracy limit) and the |x| = 8 seam between
    # the minimax polynomial and the rsqrt series.
    x = torch.tensor(
        [1e-4, -1e-4, 1e-2, -1e-2, 0.5, -0.5, 7.9, 8.0, 8.1, 12.0, -12.0],
        dtype=torch.float32,
    )

    ref = torch.special.i1e(x)
    tri = special_i1e(x)

    torch.testing.assert_close(tri, ref, rtol=5e-5, atol=2e-6)


def test_special_i1e_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_i1e")
    launches = []
    real_run = mod._i1e_flat_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._i1e_flat_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32) * 5.0
    out = mod.special_i1e(x)
    assert len(launches) >= 1, "special_i1e must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.i1e(x), rtol=5e-5, atol=2e-6)
