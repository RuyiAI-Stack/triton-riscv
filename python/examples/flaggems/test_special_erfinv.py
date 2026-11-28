import importlib

import pytest
import torch

from .special_erfinv import special_erfinv


def _tolerance(dtype):
    # Tolerance source: fp32 carries the riscv64-measured bound from the
    # erfinv_ port (1.2e-7 relative on x86_64, 7.9e-6 on the riscv64 CI
    # backend), so 5e-5 keeps a 6x margin; fp16 follows the repo
    # test_erf.py precedent (rtol=1e-2, atol=1e-3); an fp64 request runs
    # through the same narrowed path, so it reuses the fp32 bound.
    if dtype == torch.float16:
        return 1e-2, 1e-3
    return 5e-5, 5e-5


@pytest.mark.parametrize("size", [512, 1023, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
def test_special_erfinv(dtype, size):
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype).clamp(-0.999999, 0.999999)

    ref = torch.special.erfinv(x)
    tri = special_erfinv(x)

    assert tri.dtype == ref.dtype
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_special_erfinv_domain_sweep():
    # Dense sweep away from the ±1 singularity plus the band that the
    # Newton refinement dominates.
    x = torch.cat(
        [
            torch.linspace(-0.999, 0.999, 511),
            torch.linspace(-0.5, 0.5, 257),
            torch.linspace(-1e-4, 1e-4, 33),
        ]
    ).to(torch.float32)

    ref = torch.special.erfinv(x)
    tri = special_erfinv(x)
    torch.testing.assert_close(tri, ref, rtol=5e-5, atol=5e-5)


def test_special_erfinv_special_values():
    # ATen contract probed on torch CPU fp32: erfinv(±1) = ±inf,
    # |x| > 1 = nan, ±0 maps to ±0, and non-finite input stays nan.
    pos0, neg0 = 0.0, -0.0
    inf, ninf = float("inf"), float("-inf")
    nan = float("nan")
    x = torch.tensor([pos0, neg0, 1.0, -1.0, 1.5, -1.5, inf, ninf, nan])

    ref = torch.special.erfinv(x)
    tri = special_erfinv(x)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0, equal_nan=True)
    assert tri[0].item() == 0.0
    assert tri[2].item() == inf and tri[3].item() == ninf


def test_special_erfinv_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.special_erfinv")
    launches = []
    real_run = mod._special_erfinv_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._special_erfinv_kernel, "run", spy_run)
    x = torch.randn(1024, dtype=torch.float32).clamp(-0.999, 0.999)
    out = mod.special_erfinv(x)
    assert len(launches) >= 1, "special_erfinv must launch the Triton kernel"
    torch.testing.assert_close(out, torch.special.erfinv(x), rtol=5e-5, atol=5e-5)
