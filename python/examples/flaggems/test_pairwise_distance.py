import sys

import pytest
import torch
import torch.nn.functional as F

from .pairwise_distance import pairwise_distance

# Semantics (upstream flag_gems/ops/pairwise_distance.py +
# torch.nn.functional.pairwise_distance, probed on this torch 2.15.0+cpu
# build -- F.pairwise_distance is the sole reference):
# - Elementwise difference first: d = |x1 - x2 + eps| (SIGNED add, then
#   abs). Probed with eps=0.1, p=1: |0.5| -> 0.6, |-0.5| -> 0.4, |0| ->
#   0.1. This is NOT clamp_min(eps): the probed torch contract matches the
#   upstream kernel formula abs(a - b + eps) exactly.
# - Output is the p-norm of d along the last dim:
#   p=0: count of nonzero d (with eps > 0 always D, probed 4.0; with
#        eps=0 and x1 == x2 the count is 0, probed);
#   p=1: sum d; p=2: sqrt(sum d^2); p=+inf: max d; p=-inf: min d (all
#        probed); general p: (sum d^p)^(1/p) via exp2/log2.
# - Broadcasting: differing shapes go through torch broadcasting (D must
#   match); keepdim keeps the reduced dim as size 1.
# - Probed edge contracts pinned here:
#   * D == 0: finite p -> zeros over the batch shape; p=+-inf -> RuntimeError
#     (torch: "linalg.vector_norm cannot compute the inf norm ...");
#   * empty batch (0, D) -> empty output;
#   * 1-D inputs (D,) -> 0-dim output, (1,) with keepdim;
#   * float16 works on CPU torch (probed); port accumulates fp32 for fp16,
#     fp64 for fp64 (upstream acc_dtype pattern).
# - Port scope (量力拆 kernel, signature surface complete): the upstream
#   split-K two-kernel fast path (N <= 128 and D > 4096) and its 12 extra
#   kernels are not ported; the fused per-p loop kernels handle any D and
#   are functionally equivalent. libtuner/runtime configs do not exist in
#   this repo and are dropped like every other ported file.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggems package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package, which breaks re-imports by name at test runtime.
_PD_MOD = sys.modules[__name__.rsplit(".", 1)[0] + ".pairwise_distance"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call."""
    import importlib

    name = "flaggems.pairwise_distance"
    if name not in sys.modules:
        sys.modules[name] = _PD_MOD
    return importlib.import_module(name)


P_ALL = [0.0, 1.0, 2.0, 3.0, 0.5, float("inf"), float("-inf")]


@pytest.mark.parametrize("p", P_ALL)
@pytest.mark.parametrize("shape", [(8, 16), (4, 33), (2, 3, 8)])
def test_pairwise_distance_matches_torch(shape, p):
    torch.manual_seed(0)
    x1 = torch.randn(*shape)
    x2 = torch.randn(*shape)
    ref = F.pairwise_distance(x1, x2, p=p, eps=1e-6)
    out = pairwise_distance(x1, x2, p=p, eps=1e-6)
    assert out.shape == ref.shape
    assert out.dtype == x1.dtype
    if p in (0.0, float("inf"), float("-inf")):
        # Count / max / min over |diff + eps|: exact integer or single-lane
        # picks, no accumulation rounding.
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [2.0, 1.5, float("-inf")])
def test_pairwise_distance_fp64(p):
    torch.manual_seed(0)
    x1 = torch.randn(8, 16, dtype=torch.float64)
    x2 = torch.randn(8, 16, dtype=torch.float64)
    ref = F.pairwise_distance(x1, x2, p=p)
    out = pairwise_distance(x1, x2, p=p)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("p", [2.0, 1.0])
def test_pairwise_distance_fp16(p):
    torch.manual_seed(0)
    x1 = torch.randn(4, 16, dtype=torch.float16)
    x2 = torch.randn(4, 16, dtype=torch.float16)
    ref = F.pairwise_distance(x1, x2, p=p)
    out = pairwise_distance(x1, x2, p=p)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_pairwise_distance_eps_is_signed_add_then_abs():
    # Contract nail (probed): eps is added to the SIGNED difference before
    # the absolute value, so it shifts small diffs in either direction.
    x1 = torch.tensor([[0.5]])
    x2 = torch.tensor([[0.0]])
    out_pos = pairwise_distance(x1, x2, p=1, eps=0.1)
    out_neg = pairwise_distance(x2, x1, p=1, eps=0.1)
    out_same = pairwise_distance(x1, x1, p=1, eps=0.1)
    assert abs(out_pos.item() - 0.6) < 1e-6
    assert abs(out_neg.item() - 0.4) < 1e-6
    assert abs(out_same.item() - 0.1) < 1e-6
    for out, ref in [
        (out_pos, F.pairwise_distance(x1, x2, p=1, eps=0.1)),
        (out_neg, F.pairwise_distance(x2, x1, p=1, eps=0.1)),
        (out_same, F.pairwise_distance(x1, x1, p=1, eps=0.1)),
    ]:
        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-7)


def test_pairwise_distance_p0_eps_edges():
    # p=0 counts nonzero |diff + eps|: an exact -eps difference cancels to
    # zero and is NOT counted; eps=0 with identical inputs counts 0; any
    # eps > 0 makes every lane nonzero (count == D).
    x1 = torch.tensor([[-1e-6, 0.5]])
    x2 = torch.tensor([[0.0, 0.5]])
    out = pairwise_distance(x1, x2, p=0.0, eps=1e-6)
    assert out.item() == 1.0

    a = torch.tensor([[1.0, 2.0]])
    assert pairwise_distance(a, a, p=0.0, eps=0.0).item() == 0.0
    b = torch.tensor([[3.0, 4.0]])
    assert pairwise_distance(a, b, p=0.0, eps=0.0).item() == 2.0
    assert pairwise_distance(a, b, p=0.0, eps=1e-6).item() == 2.0
    assert pairwise_distance(a, a, p=0.0, eps=1e-6).item() == 2.0


@pytest.mark.parametrize("keepdim", [False, True])
def test_pairwise_distance_keepdim_shapes(keepdim):
    torch.manual_seed(0)
    x1 = torch.randn(2, 3, 4)
    x2 = torch.randn(2, 3, 4)
    out = pairwise_distance(x1, x2, p=2.0, keepdim=keepdim)
    ref = F.pairwise_distance(x1, x2, p=2.0, keepdim=keepdim)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("keepdim", [False, True])
def test_pairwise_distance_1d_input(keepdim):
    # 1-D inputs reduce over their only dim: 0-dim out, (1,) with keepdim.
    torch.manual_seed(0)
    x1 = torch.randn(8)
    x2 = torch.randn(8)
    out = pairwise_distance(x1, x2, p=2.0, keepdim=keepdim)
    ref = F.pairwise_distance(x1, x2, p=2.0, keepdim=keepdim)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("keepdim", [False, True])
def test_pairwise_distance_broadcast(keepdim):
    torch.manual_seed(0)
    x1 = torch.randn(2, 3, 4)
    x2 = torch.randn(4)
    out = pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=keepdim)
    ref = F.pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=keepdim)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    y1 = torch.randn(2, 8)
    y2 = torch.randn(2, 1, 8)
    out2 = pairwise_distance(y1, y2, p=1.0)
    ref2 = F.pairwise_distance(y1, y2, p=1.0)
    assert out2.shape == ref2.shape
    torch.testing.assert_close(out2, ref2, rtol=1e-5, atol=1e-5)


def test_pairwise_distance_identical_rows():
    # Repeated points: every lane is |0 + eps|, so p=2 -> sqrt(D) * eps.
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    out = pairwise_distance(x, x, p=2.0, eps=1e-6)
    ref = F.pairwise_distance(x, x, p=2.0, eps=1e-6)
    expected = (8**0.5) * 1e-6
    assert (out - expected).abs().max().item() < 1e-6
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_pairwise_distance_non_contiguous(p):
    torch.manual_seed(0)
    x1 = torch.randn(4, 16)[::2]  # (2, 16) row-strided
    x2 = torch.randn(16, 2).t()  # (2, 16) transposed
    assert not x1.is_contiguous()
    assert not x2.is_contiguous()
    out = pairwise_distance(x1, x2, p=p)
    ref = F.pairwise_distance(x1, x2, p=p)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [0.0, 2.0])
def test_pairwise_distance_empty_feature_dim_zero_output(p):
    # Probed: D == 0 gives zeros over the batch shape for finite p.
    x1 = torch.randn(3, 0)
    out = pairwise_distance(x1, x1, p=p)
    ref = F.pairwise_distance(x1, x1, p=p)
    assert out.shape == (3,)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out_k = pairwise_distance(x1, x1, p=p, keepdim=True)
    assert out_k.shape == (3, 1)


@pytest.mark.parametrize("p", [float("inf"), float("-inf")])
def test_pairwise_distance_empty_feature_dim_inf_raises(p):
    # Probed: no identity element for max/min over an empty reduction.
    x1 = torch.randn(3, 0)
    with pytest.raises(RuntimeError):
        pairwise_distance(x1, x1, p=p)


@pytest.mark.parametrize("keepdim", [False, True])
def test_pairwise_distance_empty_batch(keepdim):
    x1 = torch.randn(0, 4)
    out = pairwise_distance(x1, x1, p=2.0, keepdim=keepdim)
    ref = F.pairwise_distance(x1, x1, p=2.0, keepdim=keepdim)
    assert out.shape == ref.shape
    assert out.numel() == 0


@pytest.mark.parametrize(
    "p, kernel_name",
    [
        (2.0, "pairwise_distance_p2_kernel"),
        (1.0, "pairwise_distance_p1_kernel"),
        (0.0, "pairwise_distance_p0_kernel"),
        (float("inf"), "pairwise_distance_max_kernel"),
        (float("-inf"), "pairwise_distance_min_kernel"),
        (3.0, "pairwise_distance_general_kernel"),
    ],
)
def test_pairwise_distance_triton_path(monkeypatch, p, kernel_name):
    # Smoke: every p branch must launch its own Triton kernel exactly once.
    mod = _spy_module()

    launches = []
    kernel = getattr(mod, kernel_name)
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(0)
    x1 = torch.randn(8, 16)
    x2 = torch.randn(8, 16)
    out = mod.pairwise_distance(x1, x2, p=p)
    assert len(launches) == 1, f"p={p} must go through {kernel_name}"
    assert launches[0] == (2,), "grid (cdiv(8, BLOCK_M),)"
    torch.testing.assert_close(
        out, F.pairwise_distance(x1, x2, p=p), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("shape_call", ["empty_d", "empty_batch", "inf_empty_d"])
def test_pairwise_distance_short_circuits_no_kernel(monkeypatch, shape_call):
    # D == 0 (finite p) and empty batch return without any kernel launch.
    mod = _spy_module()

    launches = []
    for name in [
        "pairwise_distance_p2_kernel",
        "pairwise_distance_p1_kernel",
        "pairwise_distance_p0_kernel",
        "pairwise_distance_max_kernel",
        "pairwise_distance_min_kernel",
        "pairwise_distance_general_kernel",
    ]:
        real_run = getattr(mod, name).run

        def make_spy(real):
            def spy_run(*args, **kwargs):
                launches.append(name)
                return real(*args, **kwargs)

            return spy_run

        monkeypatch.setattr(getattr(mod, name), "run", make_spy(real_run))

    if shape_call == "empty_d":
        x = torch.randn(3, 0)
        assert mod.pairwise_distance(x, x, p=2.0).shape == (3,)
    elif shape_call == "empty_batch":
        x = torch.randn(0, 4)
        assert mod.pairwise_distance(x, x, p=2.0).shape == (0,)
    else:
        x = torch.randn(3, 0)
        with pytest.raises(RuntimeError):
            mod.pairwise_distance(x, x, p=float("inf"))
    assert launches == []
