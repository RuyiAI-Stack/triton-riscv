import importlib

import pytest
import torch

from ._standard_gamma import standard_gamma

# oracle: distributional — torch._standard_gamma samples from a different RNG
# stream, so per-value comparison is impossible. The pins are: shape/dtype,
# strict positivity (including the tiny-alpha clamp to the dtype minimum),
# moment estimates (mean = alpha, var = alpha), seeded-generator determinism,
# and state advance between calls (upstream _standard_gamma.py contract).


def test_standard_gamma_shape_dtype_positivity():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16):
        alpha = torch.rand((64, 32), dtype=dtype) * 5.0 + 0.5
        out = standard_gamma(alpha)
        assert out.shape == alpha.shape
        assert out.dtype == dtype
        assert torch.all(out > 0)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_standard_gamma_statistics(alpha):
    torch.manual_seed(1)
    out = standard_gamma(torch.full((10000,), alpha))
    mean = out.mean().item()
    var = out.var().item()
    assert abs(mean - alpha) < alpha * 0.2 + 0.5
    assert abs(var - alpha) < alpha * 0.3 + 1.0


def test_standard_gamma_small_alpha_mean():
    # alpha < 1 exercises the boost path; mean still tracks alpha.
    torch.manual_seed(2)
    out = standard_gamma(torch.full((50000,), 0.3))
    assert abs(out.mean().item() - 0.3) < 0.05


@pytest.mark.parametrize("alpha", [1e-4, 1e-3, 1e-2, 0.05])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_standard_gamma_tiny_alpha_strictly_positive(alpha, dtype):
    # ATen clamps the draw to the dtype's smallest positive value so tiny
    # alpha never underflows to zero (upstream kernel line 117).
    torch.manual_seed(3)
    out = standard_gamma(torch.full((4096,), alpha, dtype=dtype))
    assert out.dtype == dtype
    assert torch.all(out > 0), "tiny-alpha samples must stay strictly positive"


def test_standard_gamma_generator_reproducible():
    torch.manual_seed(4)
    alpha = torch.rand(2048) * 4.0 + 0.5
    gen1 = torch.Generator().manual_seed(12345)
    out1 = standard_gamma(alpha, generator=gen1)
    gen2 = torch.Generator().manual_seed(12345)
    out2 = standard_gamma(alpha, generator=gen2)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)


def test_standard_gamma_different_seeds_differ():
    torch.manual_seed(5)
    alpha = torch.rand(2048) * 4.0 + 0.5
    out1 = standard_gamma(alpha, generator=torch.Generator().manual_seed(1))
    out2 = standard_gamma(alpha, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(out1, out2)


def test_standard_gamma_generator_state_advances():
    torch.manual_seed(6)
    alpha = torch.rand(2048) * 4.0 + 0.5
    gen = torch.Generator().manual_seed(777)
    out1 = standard_gamma(alpha, generator=gen)
    out2 = standard_gamma(alpha, generator=gen)
    assert not torch.allclose(out1, out2)


def test_standard_gamma_empty_input():
    out = standard_gamma(torch.empty(0))
    assert out.shape == (0,)


def test_standard_gamma_rejects_non_floating():
    with pytest.raises(RuntimeError, match='"standard_gamma" not implemented'):
        standard_gamma(torch.ones(16, dtype=torch.int64))
    with pytest.raises(RuntimeError, match='"standard_gamma" not implemented'):
        standard_gamma(torch.ones(16, dtype=torch.bool))


def test_standard_gamma_rejects_float64():
    # The fp32 sampling kernel cannot preserve the fp64 tiny clamp value.
    with pytest.raises(RuntimeError, match="does not support float64"):
        standard_gamma(torch.rand(16, dtype=torch.float64) + 0.5)


def test_standard_gamma_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._standard_gamma")
    launches = []
    kernel = mod.standard_gamma_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(7)
    alpha = torch.rand(256) * 4.0 + 0.5
    out = standard_gamma(alpha)
    assert len(launches) >= 1
    assert out.shape == alpha.shape
    assert torch.all(out > 0)
