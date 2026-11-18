import importlib

import pytest
import torch

from .choose_qparams_optimized import choose_qparams_optimized

# torch reference: the public aten binding is compiled into this CPU build
# (the underscored torch._choose_qparams_optimized spelling is not).
torch_cqo = torch.choose_qparams_optimized

# Element counts where the discrete shrink decisions are numerically stable
# across reduction orders (sequential fp32 on the aten side, block-tree fp32
# in the Triton error kernel) - the same caveat the upstream test carries.
SHAPES = [(128,), (256,), (512,), (1024,), (64, 16)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_bins", [200, 100])
@pytest.mark.parametrize("ratio", [0.16, 0.5])
def test_choose_qparams_optimized(shape, n_bins, ratio):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    numel = x.numel()

    ref_max, ref_min = torch_cqo(x.view(-1), numel, n_bins, ratio, 8)
    res_max, res_min = choose_qparams_optimized(x, numel, n_bins, ratio, 8)

    assert res_max.dtype == torch.float32
    assert res_min.dtype == torch.float32
    assert res_max.shape == (1,) and res_min.shape == (1,)
    # The two outputs are the adjusted (max, min); order is part of the
    # contract (the aten schema's scale slot carries the max).
    assert res_max.item() >= res_min.item()
    torch.testing.assert_close(res_max, ref_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_min, ref_min, rtol=1e-5, atol=1e-5)


def test_choose_qparams_optimized_constant_input():
    # data_range == 0 branch: all values identical, no shrink step is taken.
    x = torch.full((512,), 1.5, dtype=torch.float32)

    ref_max, ref_min = torch_cqo(x, 512, 200, 0.16, 8)
    res_max, res_min = choose_qparams_optimized(x, 512, 200, 0.16, 8)

    assert res_max.item() == pytest.approx(1.5)
    assert res_min.item() == pytest.approx(1.5)
    torch.testing.assert_close(res_max, ref_max, rtol=0, atol=0)
    torch.testing.assert_close(res_min, ref_min, rtol=0, atol=0)


def test_choose_qparams_optimized_numel_smaller_than_tensor():
    # Only the first `numel` elements participate; the tail is ignored.
    # Shape/seed inside the decision-stable set (see the module docstring:
    # near-tie shrink decisions flip with the summation order, the same
    # caveat the upstream test carries).
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)

    ref_max, ref_min = torch_cqo(x, 256, 255, 0.16, 8)
    res_max, res_min = choose_qparams_optimized(x, 256, 255, 0.16, 8)

    torch.testing.assert_close(res_max, ref_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_min, ref_min, rtol=1e-5, atol=1e-5)
    # And the result genuinely differs from the full-tensor one.
    full_max, full_min = choose_qparams_optimized(x, 512, 255, 0.16, 8)
    assert res_max.item() != full_max.item()


def test_choose_qparams_optimized_non_contiguous():
    torch.manual_seed(1)
    x = torch.randn(64, 16, dtype=torch.float32).t()
    assert not x.is_contiguous()

    # aten accepts the non-contiguous view (expect_contiguous semantics);
    # probed to agree with the compacted copy.
    ref_max, ref_min = torch_cqo(x, 1024, 255, 0.16, 8)
    res_max, res_min = choose_qparams_optimized(x, 1024, 255, 0.16, 8)

    torch.testing.assert_close(res_max, ref_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_min, ref_min, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bit_width", [4, 7, 8])
def test_choose_qparams_optimized_bit_widths(bit_width):
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)

    ref_max, ref_min = torch_cqo(x, 512, 2**bit_width - 1, 0.16, bit_width)
    res_max, res_min = choose_qparams_optimized(x, 512, 2**bit_width - 1, 0.16, bit_width)

    torch.testing.assert_close(res_max, ref_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_min, ref_min, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_bins, ratio", [(1, 0.16), (255, 1.0)])
def test_choose_qparams_optimized_degenerate_bins_ratio(n_bins, ratio):
    # n_bins = 1: single shrink step spans the whole range.
    # ratio = 1.0: min_bins = 0, the guard degenerates to cur_min < cur_max.
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)

    ref_max, ref_min = torch_cqo(x, 512, n_bins, ratio, 8)
    res_max, res_min = choose_qparams_optimized(x, 512, n_bins, ratio, 8)

    torch.testing.assert_close(res_max, ref_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_min, ref_min, rtol=1e-5, atol=1e-5)


def test_choose_qparams_optimized_large_input():
    # Multi-block partial reduction (10 error blocks), beyond the upstream
    # fused kernel's residency bound. Seed inside the decision-stable set
    # (see the module docstring: near-tie shrink decisions flip with the
    # summation order).
    torch.manual_seed(2)
    x = torch.randn(10000, dtype=torch.float32)

    ref_max, ref_min = torch_cqo(x, 10000, 255, 0.16, 8)
    res_max, res_min = choose_qparams_optimized(x, 10000, 255, 0.16, 8)

    torch.testing.assert_close(res_max, ref_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_min, ref_min, rtol=1e-5, atol=1e-5)


def test_choose_qparams_optimized_deterministic():
    torch.manual_seed(0)
    x = torch.randn(512, dtype=torch.float32)

    a = choose_qparams_optimized(x, 512, 255, 0.16, 8)
    b = choose_qparams_optimized(x, 512, 255, 0.16, 8)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_choose_qparams_optimized_rejects_non_float32():
    # aten: RuntimeError "expected scalar type Float but found Double";
    # the port keeps upstream's assert guard.
    x = torch.randn(8, dtype=torch.float64)
    with pytest.raises(AssertionError):
        choose_qparams_optimized(x, 8, 255, 0.16, 8)
    with pytest.raises(RuntimeError):
        torch_cqo(x, 8, 255, 0.16, 8)


def test_choose_qparams_optimized_empty_input():
    # aten raises ValueError("input tensor is empty and has no data"); the
    # port mirrors the guard instead of returning search garbage (the
    # upstream module is undefined behaviour here).
    x = torch.zeros(0, dtype=torch.float32)
    with pytest.raises(ValueError):
        choose_qparams_optimized(x, 0, 255, 0.16, 8)
    with pytest.raises(ValueError):
        torch_cqo(x, 0, 255, 0.16, 8)


def test_choose_qparams_optimized_triton_path(monkeypatch):
    # Smoke: one call must drive the min/max kernel, at least one error
    # evaluation and exactly one final store, all through Triton.
    mod = importlib.import_module("flaggems.choose_qparams_optimized")

    counts = {"minmax": 0, "qerr": 0, "store": 0}

    def spy(name, real):
        def spy_run(*args, **kwargs):
            counts[name] += 1
            return real(*args, **kwargs)

        return spy_run

    monkeypatch.setattr(
        mod._minmax_kernel, "run", spy("minmax", mod._minmax_kernel.run)
    )
    monkeypatch.setattr(
        mod._qerror_kernel, "run", spy("qerr", mod._qerror_kernel.run)
    )
    monkeypatch.setattr(
        mod._store2_kernel, "run", spy("store", mod._store2_kernel.run)
    )

    torch.manual_seed(0)
    x = torch.randn(256, dtype=torch.float32)
    res_max, res_min = mod.choose_qparams_optimized(x, 256, 255, 0.16, 8)

    assert counts["minmax"] == 1
    assert counts["qerr"] >= 1
    assert counts["store"] == 1
    ref_max, ref_min = torch_cqo(x, 256, 255, 0.16, 8)
    torch.testing.assert_close(res_max, ref_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_min, ref_min, rtol=1e-5, atol=1e-5)
