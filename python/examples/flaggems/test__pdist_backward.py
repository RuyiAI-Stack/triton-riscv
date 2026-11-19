import sys

import pytest
import torch

from ._pdist_backward import _pdist_backward
from ._pdist_forward import _pdist_forward

# Semantics (upstream flag_gems/ops/_pdist_backward.py + torch autograd,
# probed on this torch 2.15.0+cpu build):
# - Signature matches aten::_pdist_backward(grad_output, self, p, pdist);
#   pdist is the saved forward output. float32 only (upstream assert; the
#   reference is torch.autograd over torch.pdist on the same grad vector).
# - Per-p gradient of dist(i,j) w.r.t. the two rows, accumulated over all
#   pairs a row participates in (both loop directions j < i and j > i):
#   * p=2: g * (xi - xj) / dist, contribution 0 when dist == 0
#     (probed: identical rows contribute nothing, no NaN).
#   * p=1: g * sign(xi - xj), 0 where the coordinate difference is 0
#     (probed: grad [[-1,-1],[-1,2],[2,-1]] for integer-valued input).
#   * p=inf: g * sign * [|diff| == dist]; tied maxima each get the FULL g
#     (probed: [[1,-1,0.5],[-1,1,0.3]] -> grad [[1,-1,0],[-1,1,0]],
#     no even splitting). Upstream matches.
#   * general p: g * sign * (|diff|/dist)^(p-1), 0 where diff == 0 or
#     dist == 0 (probed to match the manual fp64 formula bit-exactly).
#   * p=0: torch returns exact zeros (probed); upstream would route p=0
#     into its general kernel and produce garbage (ratio^(-1)), so this
#     port returns zeros on the host without launching a kernel.
# - N == 0 returns an empty (0, M) tensor without launching; N == 1
#   launches (loops are empty) and writes zeros, matching torch autograd
#   of an empty pdist output.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggems package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package, which breaks re-imports by name at test runtime.
_PB_MOD = sys.modules[__name__.rsplit(".", 1)[0] + "._pdist_backward"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call."""
    import importlib

    name = "flaggems._pdist_backward"
    if name not in sys.modules:
        sys.modules[name] = _PB_MOD
    return importlib.import_module(name)


def _autograd_ref(x, p, g):
    xa = x.clone().requires_grad_(True)
    (torch.pdist(xa, p) * g).sum().backward()
    return xa.grad


P_GRAD = [1.0, 2.0, 1.5, 3.0, 0.5, float("inf")]


@pytest.mark.parametrize("p", P_GRAD)
@pytest.mark.parametrize("N, M", [(8, 16), (6, 32)])
def test__pdist_backward_matches_autograd(N, M, p):
    torch.manual_seed(0)
    x = torch.randn(N, M)
    g = torch.randn(N * (N - 1) // 2)
    pd = torch.pdist(x, p)

    out = _pdist_backward(g, x, p, pd)
    ref = _autograd_ref(x, p, g)

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    if p == float("inf"):
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test__pdist_backward_p0_returns_zeros(monkeypatch):
    # Probed: torch autograd of pdist(x, 0) is exactly zero everywhere.
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    g = torch.randn(10)
    pd = torch.pdist(x, 0.0)

    out = _pdist_backward(g, x, 0.0, pd)
    assert torch.equal(out, torch.zeros_like(x))
    ref = _autograd_ref(x, 0.0, g)
    assert torch.equal(ref, torch.zeros_like(x))


def test__pdist_backward_inf_tied_maxima():
    # Two coordinates tie at the max |diff| (2.0): each gets the full g with
    # its sign, no splitting (probed against autograd).
    x = torch.tensor([[1.0, -1.0, 0.5], [-1.0, 1.0, 0.3], [1.0, -1.0, 0.5]])
    g = torch.tensor([0.7, -1.2, 3.0])
    pd = torch.pdist(x, float("inf"))

    out = _pdist_backward(g, x, float("inf"), pd)
    ref = _autograd_ref(x, float("inf"), g)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test__pdist_backward_p2_identical_rows():
    # dist == 0 contributes exactly 0 (probed: no NaN, remaining pairs give
    # the +/- 1/sqrt(3) pattern).
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    g = torch.ones(3)
    pd = torch.pdist(x, 2.0)

    out = _pdist_backward(g, x, 2.0, pd)
    ref = _autograd_ref(x, 2.0, g)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test__pdist_backward_p1_zero_diff_coords():
    # Integer-valued input: sign pattern with 0 at matching coordinates
    # (probed grad [[-1,-1],[-1,2],[2,-1]]).
    x = torch.tensor([[1.0, 2.0], [1.0, 3.0], [5.0, 2.0]])
    g = torch.ones(3)
    pd = torch.pdist(x, 1.0)

    out = _pdist_backward(g, x, 1.0, pd)
    ref = _autograd_ref(x, 1.0, g)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test__pdist_backward_general_p_zero_diff_coords():
    # Coordinate with diff == 0 gets gradient 0 under general p (probed
    # fp64: [[0, -1], [0, 1]] for g=1 on [[1, 2], [1, 3]] with p=1.5).
    x = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    g = torch.ones(1)
    pd = torch.pdist(x, 1.5)

    out = _pdist_backward(g, x, 1.5, pd)
    ref = _autograd_ref(x, 1.5, g)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    assert out[0, 0].item() == 0.0


def test__pdist_backward_roundtrip_with_forward():
    # Composition nail: our own forward output feeds our backward, and the
    # result still matches torch autograd.
    torch.manual_seed(0)
    x = torch.randn(7, 12)
    g = torch.randn(21)

    out = _pdist_backward(g, x, 2.0, _pdist_forward(x, 2.0))
    ref = _autograd_ref(x, 2.0, g)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("p", [1.0, 2.0])
def test__pdist_backward_non_contiguous_inputs(p):
    torch.manual_seed(0)
    x = torch.randn(16, 8).t()  # (8, 16) non-contiguous
    n_pairs = 8 * 7 // 2
    g_dense = torch.randn(n_pairs)
    pd_dense = torch.pdist(x, p)
    # Interleave then slice: values preserved, memory non-contiguous.
    g = torch.stack([g_dense, torch.zeros(n_pairs)], dim=1).t()[0]
    pd = torch.stack([pd_dense, torch.zeros(n_pairs)], dim=1).t()[0]
    assert not g.is_contiguous()
    assert not pd.is_contiguous()
    assert not x.is_contiguous()

    out = _pdist_backward(g, x, p, pd)
    ref = _autograd_ref(x, p, g_dense)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test__pdist_backward_single_row_zeros():
    # N=1: no pairs, torch autograd yields zeros for the row.
    torch.manual_seed(0)
    x = torch.randn(1, 6)
    g = torch.zeros(0)
    pd = torch.pdist(x, 2.0)

    out = _pdist_backward(g, x, 2.0, pd)
    assert out.shape == (1, 6)
    assert torch.equal(out, torch.zeros_like(x))
    ref = _autograd_ref(x, 2.0, g)
    assert torch.equal(ref, torch.zeros_like(x))


def test__pdist_backward_empty_rows():
    x = torch.randn(0, 6)
    g = torch.zeros(0)
    pd = torch.pdist(x, 2.0)
    out = _pdist_backward(g, x, 2.0, pd)
    assert out.shape == (0, 6)


def test__pdist_backward_dtype_and_ndim_asserts():
    x = torch.randn(4, 8, dtype=torch.float64)
    with pytest.raises(AssertionError):
        _pdist_backward(torch.zeros(6), x, 2.0, torch.pdist(x, 2.0))
    with pytest.raises(AssertionError):
        _pdist_backward(torch.zeros(6), torch.randn(2, 3, 4), 2.0, None)


@pytest.mark.parametrize(
    "p, kernel_name",
    [
        (2.0, "_pdist_backward_p2_kernel"),
        (1.0, "_pdist_backward_p1_kernel"),
        (float("inf"), "_pdist_backward_inf_kernel"),
        (1.5, "_pdist_backward_general_kernel"),
    ],
)
def test__pdist_backward_triton_path(monkeypatch, p, kernel_name):
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
    x = torch.randn(8, 16)
    g = torch.randn(28)
    out = mod._pdist_backward(g, x, p, torch.pdist(x, p))
    assert len(launches) == 1, f"p={p} must go through {kernel_name}"
    assert launches[0] == (1, 8), "grid (cdiv(16, BLOCK_M), N)"
    ref = _autograd_ref(x, p, g)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test__pdist_backward_p0_no_kernel(monkeypatch):
    # p=0 returns host zeros without any kernel launch.
    mod = _spy_module()

    launches = []
    for name in [
        "_pdist_backward_p2_kernel",
        "_pdist_backward_p1_kernel",
        "_pdist_backward_inf_kernel",
        "_pdist_backward_general_kernel",
    ]:
        real_run = getattr(mod, name).run

        def make_spy(real):
            def spy_run(*args, **kwargs):
                launches.append(name)
                return real(*args, **kwargs)

            return spy_run

        monkeypatch.setattr(getattr(mod, name), "run", make_spy(real_run))

    x = torch.randn(4, 8)
    out = mod._pdist_backward(torch.randn(6), x, 0.0, torch.pdist(x, 0.0))
    assert torch.equal(out, torch.zeros_like(x))
    assert launches == []
