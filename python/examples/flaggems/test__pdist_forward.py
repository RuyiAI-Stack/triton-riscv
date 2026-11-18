import sys

import pytest
import torch

from ._pdist_forward import _pdist_forward

# Semantics (upstream flag_gems/ops/_pdist_forward.py + torch.pdist, probed on
# this torch 2.15.0+cpu build):
# - x is (N, M): N points of M coordinates. There is no batch dim: pdist
#   accepts exactly 2D input (aten contract, kept via assert).
# - Output is (N*(N-1)/2,) in row-major upper-triangle order: the pair (i, j),
#   i < j, lands at index i*N - i*(i+1)//2 + (j - i - 1).
# - p=2: sqrt(sum diff^2); p=1: sum |diff|; p=0: count of non-matching
#   coordinates (probed: torch.pdist(x, 0) == 3.0 for 3 nonzero diffs);
#   p=inf: max |diff|; general p: (sum |diff|^p)^(1/p).
# - Probed torch contracts this port pins:
#   * p < 0 -> RuntimeError "pdist only supports non-negative p values"
#     (upstream dispatches negative p to the general kernel; fixed here).
#   * N < 2 -> empty (0,) output, no kernel launch.
#   * M == 0 -> all-zero (n_pairs,) output for EVERY p including inf (probed;
#     differs from pairwise_distance, which raises for inf on empty D).
#     Upstream would crash here (triton.next_power_of_2(0) == 0), so the
#     wrapper short-circuits.
#   * float16 -> NotImplementedError on torch; the port keeps the upstream
#     FlagGems assert (AssertionError) like the _conj port kept upstream's
#     RuntimeError contract.
#   * Only float32/float64 inputs; non-contiguous inputs handled via
#     .contiguous() (probed: torch.pdist accepts them, same values).
# - Upstream divergences fixed against torch: p=0 upstream falls into the
#   general kernel and returns exp(log(count)/0) = inf; this port has an
#   explicit counting kernel (p0). Accumulation runs in the input dtype's
#   precision (float64 in / float64 acc), upstream hard-codes fp32 math.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggems package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package, which breaks re-imports by name at test runtime.
_PF_MOD = sys.modules[__name__.rsplit(".", 1)[0] + "._pdist_forward"]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call."""
    import importlib

    name = "flaggems._pdist_forward"
    if name not in sys.modules:
        sys.modules[name] = _PF_MOD
    return importlib.import_module(name)


P_ALL = [0.0, 1.0, 2.0, 3.0, 0.5, float("inf")]


def _tol(dtype):
    if dtype == torch.float64:
        return 1e-9, 1e-9
    return 1e-5, 1e-5


@pytest.mark.parametrize("p", P_ALL)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("N, M", [(8, 16), (10, 8)])
def test__pdist_forward_matches_torch(N, M, dtype, p):
    torch.manual_seed(0)
    x = torch.randn(N, M, dtype=dtype)
    ref = torch.pdist(x, p)
    out = _pdist_forward(x, p)
    assert out.shape == (N * (N - 1) // 2,)
    assert out.dtype == dtype
    if p == 0.0:
        # Integer-valued counts: exact.
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    else:
        rtol, atol = _tol(dtype)
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test__pdist_forward_pair_ordering():
    # Integer-valued input, p=1: distances are exact integers, so the pair
    # ordering (i, j) -> i*N - i*(i+1)//2 + (j-i-1) is pinned bit-exactly.
    N, M = 4, 3
    x = torch.arange(N * M, dtype=torch.float32).reshape(N, M)
    ref = torch.pdist(x, 1.0)
    manual = torch.tensor(
        [
            (x[i] - x[j]).abs().sum()
            for i in range(N)
            for j in range(i + 1, N)
        ]
    )
    torch.testing.assert_close(ref, manual, rtol=0, atol=0)
    out = _pdist_forward(x, 1.0)
    torch.testing.assert_close(out, manual, rtol=0, atol=0)


@pytest.mark.parametrize("p", [1.0, 2.0, float("inf")])
def test__pdist_forward_repeated_rows(p):
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    x[1] = x[0]
    x[3] = x[2]
    ref = torch.pdist(x, p)
    out = _pdist_forward(x, p)
    rtol, atol = _tol(torch.float32)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    # Pairs (0,1) and (2,3) sit at indices 0 and 5 for N=4.
    assert out[0].item() == 0.0
    assert out[5].item() == 0.0


def test__pdist_forward_p0_counts_repeated_rows():
    # p=0 counts non-matching coordinates: identical rows -> 0, one differing
    # coordinate -> 1 (probed torch.pdist values [0., 1., 1.]).
    x = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 3.0]])
    out = _pdist_forward(x, 0.0)
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0, 1.0]), rtol=0, atol=0)


@pytest.mark.parametrize("N", [0, 1])
def test__pdist_forward_few_rows_empty(N):
    x = torch.randn(N, 4)
    out = _pdist_forward(x, 2.0)
    ref = torch.pdist(x, 2.0)
    assert out.shape == (0,)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("p", [0.0, 2.0, float("inf")])
def test__pdist_forward_empty_feature_dim(p):
    # Probed: torch.pdist on (3, 0) returns zeros(3) for every p, including
    # inf (unlike pairwise_distance on empty D, which raises).
    x = torch.randn(3, 0)
    out = _pdist_forward(x, p)
    ref = torch.pdist(x, p)
    assert out.shape == (3,)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("p", [1.0, 2.0])
def test__pdist_forward_non_contiguous(p):
    torch.manual_seed(0)
    x = torch.randn(16, 8).t()  # (8, 16) non-contiguous
    assert not x.is_contiguous()
    out = _pdist_forward(x, p)
    ref = torch.pdist(x, p)
    rtol, atol = _tol(torch.float32)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def test__pdist_forward_negative_p_raises():
    x = torch.randn(3, 4)
    with pytest.raises(RuntimeError, match="non-negative"):
        _pdist_forward(x, -1.0)
    with pytest.raises(RuntimeError, match="non-negative"):
        _pdist_forward(x, float("-inf"))


def test__pdist_forward_dtype_and_ndim_asserts():
    x = torch.randn(3, 4)
    # torch raises NotImplementedError for fp16; the port keeps the upstream
    # FlagGems dtype assert (AssertionError), like the _conj port convention.
    with pytest.raises(AssertionError):
        _pdist_forward(x.to(torch.float16), 2.0)
    with pytest.raises(AssertionError):
        _pdist_forward(torch.randn(2, 3, 4), 2.0)


@pytest.mark.parametrize(
    "p, kernel_name",
    [
        (2.0, "_pdist_forward_p2_kernel"),
        (1.0, "_pdist_forward_p1_kernel"),
        (0.0, "_pdist_forward_p0_kernel"),
        (float("inf"), "_pdist_forward_inf_kernel"),
        (3.0, "_pdist_forward_general_kernel"),
    ],
)
def test__pdist_forward_triton_path(monkeypatch, p, kernel_name):
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
    out = mod._pdist_forward(x, p)
    assert len(launches) == 1, f"p={p} must go through {kernel_name}"
    assert launches[0] == (28,), "N=8 -> 28 pairs"
    torch.testing.assert_close(out, torch.pdist(x, p), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [2.0, float("inf")])
def test__pdist_forward_short_circuits_no_kernel(monkeypatch, p):
    # N < 2 and M == 0 must return without any kernel launch.
    mod = _spy_module()

    launches = []
    real_runs = {
        name: getattr(mod, name).run
        for name in [
            "_pdist_forward_p2_kernel",
            "_pdist_forward_p1_kernel",
            "_pdist_forward_p0_kernel",
            "_pdist_forward_inf_kernel",
            "_pdist_forward_general_kernel",
        ]
    }

    def make_spy(name, real):
        def spy_run(*args, **kwargs):
            launches.append(name)
            return real(*args, **kwargs)

        return spy_run

    for name, real in real_runs.items():
        monkeypatch.setattr(getattr(mod, name), "run", make_spy(name, real))

    assert _pdist_forward(torch.randn(1, 4), p).shape == (0,)
    assert _pdist_forward(torch.randn(3, 0), p).shape == (3,)
    assert launches == []
