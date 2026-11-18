import pytest
import torch

from .binary_cross_entropy_backward import binary_cross_entropy_backward


def _autograd_ref(grad_output, p, t, weight, reduction):
    # Torch reference: differentiate the "none"-reduced BCE elementwise and
    # scale afterwards, which composes exactly like aten for all three
    # reductions (mean divides by the element count).
    x = p.clone().requires_grad_()
    loss = torch.nn.functional.binary_cross_entropy(
        x, t, weight=weight, reduction="none"
    )
    # autograd.grad does not broadcast a 0-dim grad_outputs, so expand
    # scalar grad_output explicitly (mathematically identical).
    if grad_output.dim() == 0:
        grad_output = grad_output.expand(loss.shape)
    (g,) = torch.autograd.grad(loss, x, grad_outputs=grad_output)
    if reduction == "mean":
        return g / p.numel()
    if reduction == "sum":
        return g
    return g


def _safe_prob(size, dtype=torch.float32, lo=0.05, hi=0.95):
    return torch.rand(size, dtype=dtype).clamp(lo, hi)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_bce_backward(reduction, size):
    torch.manual_seed(0)
    p = _safe_prob(size)
    t = torch.rand(size)
    g = torch.randn(size)

    tri = binary_cross_entropy_backward(g, p, t, None, reduction)
    ref = _autograd_ref(g, p, t, None, reduction)

    assert tri.shape == p.shape
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("size", [512, 1023])
def test_bce_backward_weight(reduction, size):
    torch.manual_seed(0)
    p = _safe_prob(size)
    t = torch.rand(size)
    g = torch.randn(size)
    w = torch.rand(size) + 0.5

    tri = binary_cross_entropy_backward(g, p, t, w, reduction)
    ref = _autograd_ref(g, p, t, w, reduction)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_bce_backward_scalar_grad_output():
    # aten broadcasts a scalar grad_output over the grad_input.
    torch.manual_seed(0)
    p = _safe_prob(1024)
    t = torch.rand(1024)
    g = torch.tensor(0.5)

    for reduction in ("none", "mean", "sum"):
        tri = binary_cross_entropy_backward(g, p, t, None, reduction)
        ref = _autograd_ref(g, p, t, None, reduction)
        torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_bce_backward_broadcast():
    torch.manual_seed(0)
    p = _safe_prob((4, 128))
    t = torch.rand(4, 128)
    g = torch.randn(4, 128)
    w = torch.rand(1, 128) + 0.5

    tri = binary_cross_entropy_backward(g, p, t, w, "none")
    ref = _autograd_ref(g, p, t, w, "none")
    assert tri.shape == (4, 128)
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        # fp16/bf16: the kernel computes in fp32 and narrows once; the
        # fp16 ulp near |grad| ~ 10 is ~0.0078, so 1e-2 covers one ulp.
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_bce_backward_half_dtypes(dtype, rtol, atol):
    torch.manual_seed(0)
    p = _safe_prob(512, dtype, lo=0.1, hi=0.9)
    t = torch.rand(512, dtype=dtype)
    g = torch.randn(512, dtype=dtype)

    tri = binary_cross_entropy_backward(g, p, t, None, "none")
    ref = _autograd_ref(g, p, t, None, "none")
    torch.testing.assert_close(tri, ref, rtol=rtol, atol=atol)


def test_bce_backward_denominator_clamp():
    # FlagGems semantics (upstream): the denominator p*(1-p) is clamped to
    # 1e-6, so p at (or numerically at) 0/1 stays finite -- torch itself
    # returns inf/0/NaN there, hence the direct formula comparison.
    p = torch.tensor([0.0, 1.0, 1e-7, 1.0 - 1e-7, 0.5])
    t = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5])
    g = torch.ones(5)
    w = torch.tensor([2.0, 0.5, 1.0, 1.0, 1.0])

    for weight in (None, w):
        tri = binary_cross_entropy_backward(g, p, t, weight, "none")
        expected = g * (p - t) / torch.clamp(p * (1 - p), min=1e-6)
        if weight is not None:
            expected = expected * weight
        torch.testing.assert_close(tri, expected, rtol=1e-5, atol=1e-6)


def test_bce_backward_int_reduction_enum():
    # aten-style enum 0/1/2 is accepted alongside the strings
    # (repo smooth_l1_loss convention).
    torch.manual_seed(0)
    p = _safe_prob(512)
    t = torch.rand(512)
    g = torch.randn(512)

    by_enum = [
        binary_cross_entropy_backward(g, p, t, None, k) for k in (0, 1, 2)
    ]
    for reduction, enum_result in zip(("none", "mean", "sum"), by_enum):
        torch.testing.assert_close(
            enum_result,
            binary_cross_entropy_backward(g, p, t, None, reduction),
            rtol=0.0,
            atol=0.0,
        )


def test_bce_backward_fp64_aten_fallback(monkeypatch):
    # fp64 goes through torch.ops.aten.binary_cross_entropy_backward (full
    # double precision).  Spy on the aten op to assert the fallback is the
    # code path actually taken.
    real = torch.ops.aten.binary_cross_entropy_backward
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(
        torch.ops.aten, "binary_cross_entropy_backward", spy
    )

    torch.manual_seed(0)
    p = _safe_prob(128, dtype=torch.float64)
    t = torch.rand(128, dtype=torch.float64)
    g = torch.randn(128, dtype=torch.float64)
    w = torch.rand(128, dtype=torch.float64) + 0.5

    for reduction in ("none", "mean", "sum"):
        tri = binary_cross_entropy_backward(g, p, t, w, reduction)
        ref = _autograd_ref(g, p, t, w, reduction)
        torch.testing.assert_close(tri, ref, rtol=1e-12, atol=1e-12)

    assert calls, "expected the fp64 path to call the aten op"
    assert len(calls) == 3


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_bce_backward_empty(shape, reduction):
    torch.manual_seed(0)
    p = torch.rand(shape)
    t = torch.rand(shape)
    g = torch.ones(shape)

    tri = binary_cross_entropy_backward(g, p, t, None, reduction)

    assert tri.shape == shape
    ref = _autograd_ref(g, p, t, None, reduction)
    assert tri.shape == ref.shape


def test_bce_backward_non_contiguous():
    torch.manual_seed(0)
    p = _safe_prob((64, 32)).t()
    t = torch.rand(64, 32).t()
    g = torch.randn(64, 32).t()
    assert not p.is_contiguous() and not g.is_contiguous()

    tri = binary_cross_entropy_backward(g, p, t, None, "none")
    ref = _autograd_ref(g, p, t, None, "none")
    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


def test_bce_backward_integer_input_rejected():
    p = torch.randint(0, 2, (8,))
    t = p.clone()
    g = torch.ones(8)
    with pytest.raises(RuntimeError, match="floating-point"):
        binary_cross_entropy_backward(g, p, t)


def test_bce_backward_dtype_mismatch_rejected():
    torch.manual_seed(0)
    p = _safe_prob(8)
    t64 = torch.rand(8, dtype=torch.float64)
    g = torch.ones(8)
    with pytest.raises(RuntimeError, match="Found dtype"):
        binary_cross_entropy_backward(g, p, t64)


def test_bce_backward_invalid_reduction():
    torch.manual_seed(0)
    p = _safe_prob(8)
    t = torch.rand(8)
    g = torch.ones(8)
    with pytest.raises(ValueError, match="reduction"):
        binary_cross_entropy_backward(g, p, t, None, "max")


def test_bce_backward_triton_path(monkeypatch):
    # Smoke: the unweighted fp32 "none" path must launch the Triton kernel.
    import importlib

    mod = importlib.import_module("flaggems.binary_cross_entropy_backward")

    launches = []
    real_run = mod._binary_cross_entropy_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._binary_cross_entropy_backward_kernel, "run", spy_run)
    p = _safe_prob(1024)
    t = torch.rand(1024)
    g = torch.randn(1024)
    out = mod.binary_cross_entropy_backward(g, p, t, None, "none")
    assert len(launches) == 1, "the main path must go through the Triton kernel"
    torch.testing.assert_close(
        out, _autograd_ref(g, p, t, None, "none"), rtol=1e-4, atol=1e-4
    )
