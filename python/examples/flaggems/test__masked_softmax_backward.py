import sys

import pytest
import torch
import torch.nn.functional as F

from ._masked_softmax import _masked_softmax
from ._masked_softmax_backward import _masked_softmax_backward

# Semantics (upstream flag_gems/ops/_masked_softmax_backward.py + aten, probed
# on this host with torch 2.15.0a0+cpu): _masked_softmax_backward(grad_output,
# output, mask, dim=None) is the grad of _masked_softmax with respect to the
# input, with a same-shape elementwise mask (the mask_type-2 form; aten has
# no mask_type here).  For every reduction row:
#     out'   = where(mask, 0, output)          # masked lanes carry no grad
#     scale  = sum_k out'[k] * grad_output[k]
#     grad   = where(mask, 0, out' * (grad_output - scale))
# Probed equal (fp64, 1e-12) to torch.autograd.grad of
# F.softmax(x.masked_fill(mask, -inf), dim) w.r.t. x: masked_fill zeroes the
# gradient at masked positions and fully-masked rows backprop to exactly 0
# (their forward is NaN but the backward never looks at those lanes -- the
# kernel zero-reads `output` at masked positions BEFORE using it, so a raw
# forward output with NaN fully-masked rows also produces 0 gradients,
# matching aten, probed).  dim defaults to the last dim; negative dims are
# normalized; out-of-range dims are rejected with an assert (upstream text,
# while aten raises IndexError).  The mask may be bool or any integer dtype
# compared against 0 (upstream kernel loads it as `m != 0`, no dtype assert
# unlike the forward).  Output dtype == output dtype.
#
# Tolerances: kernel accumulates the dot product in fp32 like torch's CPU
# softmax backward, different summation order -> few-ulp drift.  fp32 uses
# rtol=atol=1e-4 (repo softmax-family convention); fp16 rounds once from
# the fp32 tile -> rtol=atol=1e-3.

# The op module object, captured at collection time. pytest 9.2 evicts the
# flaggems package from sys.modules after collection, and this environment's
# FileFinder cache for python/examples nondeterministically forgets the
# package (probed: package absent from FileFinder._path_cache while present
# on disk), which breaks re-imports by name at test runtime.
_MASKED_SOFTMAX_BACKWARD_MOD = sys.modules[
    __name__.rsplit(".", 1)[0] + "._masked_softmax_backward"
]


def _spy_module():
    """Fetch the op module via importlib (S1 rule: package __init__ exports
    shadow submodule attributes). Re-seed sys.modules from the object the
    collection-time import kept alive before the mandated import_module call
    (a direct sys.modules hit needs no parent lookup)."""
    import importlib

    name = "flaggems._masked_softmax_backward"
    if name not in sys.modules:
        sys.modules[name] = _MASKED_SOFTMAX_BACKWARD_MOD
    return importlib.import_module(name)


def _forward_ref(x, mask, dim):
    # aten-contract forward output: masked positions exactly 0.0 (fully
    # masked rows too -- their NaN lanes are all masked).
    out = F.softmax(x.masked_fill(mask, float("-inf")), dim)
    return torch.where(mask, torch.zeros_like(out), out)


def _grad_ref(x, mask, dim, grad_output):
    x = x.detach().clone().requires_grad_(True)
    return torch.autograd.grad(
        F.softmax(x.masked_fill(mask, float("-inf")), dim), x, grad_output
    )[0]


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_masked_softmax_backward_2d(size):
    torch.manual_seed(0)
    x = torch.randn(size, size, dtype=torch.float32)
    mask = torch.rand(size, size) < 0.3
    grad_output = torch.randn(size, size, dtype=torch.float32)
    output = _forward_ref(x, mask, -1)

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    assert grad_input.dtype == torch.float32
    assert grad_input.shape == output.shape
    torch.testing.assert_close(
        grad_input, _grad_ref(x, mask, -1, grad_output), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("dim", [-1, 0, 1, 2, -2])
def test_masked_softmax_backward_dim(dim):
    # Every reduction axis (including negatives); non-last dims exercise
    # the strided (inner > 1) kernel path.
    torch.manual_seed(0)
    x = torch.randn(8, 16, 32, dtype=torch.float32)
    mask = torch.rand(8, 16, 32) < 0.4
    grad_output = torch.randn(8, 16, 32, dtype=torch.float32)
    output = _forward_ref(x, mask, dim)

    grad_input = _masked_softmax_backward(grad_output, output, mask, dim)

    torch.testing.assert_close(
        grad_input, _grad_ref(x, mask, dim, grad_output), rtol=1e-4, atol=1e-4
    )


def test_masked_softmax_backward_dim_default_is_last():
    torch.manual_seed(0)
    x = torch.randn(16, 64, dtype=torch.float32)
    mask = torch.rand(16, 64) < 0.5
    grad_output = torch.randn(16, 64, dtype=torch.float32)
    output = _forward_ref(x, mask, -1)

    grad_default = _masked_softmax_backward(grad_output, output, mask, None)
    grad_last = _masked_softmax_backward(grad_output, output, mask, -1)

    torch.testing.assert_close(grad_default, grad_last, rtol=1e-4, atol=1e-4)


def test_masked_softmax_backward_masked_positions_exactly_zero():
    torch.manual_seed(0)
    x = torch.randn(16, 64, dtype=torch.float32)
    mask = torch.rand(16, 64) < 0.5
    grad_output = torch.randn(16, 64, dtype=torch.float32)
    output = _forward_ref(x, mask, -1)

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    assert (grad_input[mask] == 0).all(), "masked positions must carry no grad"


def test_masked_softmax_backward_fully_masked_row_zero_grad():
    # Fully-masked rows backprop to exactly 0 (autograd through
    # masked_fill(-inf) drops those lanes; the kernel zero-reads the NaN
    # output lanes before using them).
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    mask[3] = True
    grad_output = torch.randn(8, 64, dtype=torch.float32)
    output = _forward_ref(x, mask, -1)

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    assert (grad_input[3] == 0).all()
    ref = _grad_ref(x, mask, -1, grad_output)
    torch.testing.assert_close(grad_input[:3], ref[:3], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_input[4:], ref[4:], rtol=1e-4, atol=1e-4)


def test_masked_softmax_backward_nan_output_fully_masked_row():
    # Raw forward output (NaN fully-masked rows, NOT pre-zeroed) must still
    # produce 0 gradients there -- the kernel masks the loaded output.
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    mask[3] = True
    grad_output = torch.randn(8, 64, dtype=torch.float32)
    output = F.softmax(x.masked_fill(mask, float("-inf")), -1)  # raw, NaN row
    assert torch.isnan(output[3]).all()

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    assert not torch.isnan(grad_input).any()
    assert (grad_input[3] == 0).all()
    ref = _grad_ref(x, mask, -1, grad_output)
    torch.testing.assert_close(grad_input, ref, rtol=1e-4, atol=1e-4)


def test_masked_softmax_backward_matches_forward_output():
    # Integration: grads computed from this port's own forward output agree
    # with the autograd reference.
    torch.manual_seed(0)
    x = torch.randn(32, 128, dtype=torch.float32)
    mask = torch.rand(32, 128) < 0.4
    grad_output = torch.randn(32, 128, dtype=torch.float32)
    output = _masked_softmax(x, mask, -1, 2)
    output = torch.where(mask, torch.zeros_like(output), output)

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    torch.testing.assert_close(
        grad_input, _grad_ref(x, mask, -1, grad_output), rtol=1e-4, atol=1e-4
    )


def test_masked_softmax_backward_matches_aten():
    # Oracle cross-check (probed: aten CPU == the where-formula == autograd).
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    grad_output = torch.randn(8, 64, dtype=torch.float32)
    output = _forward_ref(x, mask, -1)

    aten_grad = torch.ops.aten._masked_softmax_backward(grad_output, output, mask, -1)

    torch.testing.assert_close(
        _masked_softmax_backward(grad_output, output, mask, -1),
        aten_grad,
        rtol=1e-4, atol=1e-4,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_masked_softmax_backward_dtype_matrix(dtype):
    torch.manual_seed(0)
    x = torch.randn(8, 512, dtype=torch.float32).to(dtype)
    mask = torch.rand(8, 512) < 0.5
    grad_output = torch.randn(8, 512, dtype=torch.float32).to(dtype)
    output = _forward_ref(x, mask, -1)

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    assert grad_input.dtype == dtype
    tol = 1e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(
        grad_input, _grad_ref(x, mask, -1, grad_output), rtol=tol, atol=tol
    )


def test_masked_softmax_backward_uint8_mask():
    # The kernel compares the mask against 0 (no bool assert in the
    # backward), so integer masks work like _masked_scale's contract.
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.tensor([5, 0, -3, 2] * 16 * 8, dtype=torch.int32).reshape(8, 64)
    grad_output = torch.randn(8, 64, dtype=torch.float32)
    output = _forward_ref(x, mask != 0, -1)

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    torch.testing.assert_close(
        grad_input, _grad_ref(x, mask != 0, -1, grad_output), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("shape", [(0, 512), (512, 0)])
def test_masked_softmax_backward_empty(shape):
    grad_output = torch.randn(shape, dtype=torch.float32)
    output = torch.randn(shape, dtype=torch.float32)
    mask = torch.rand(shape) < 0.5

    grad_input = _masked_softmax_backward(grad_output, output, mask, -1)

    assert grad_input.shape == output.shape
    assert grad_input.dtype == output.dtype
    assert grad_input.numel() == 0


def test_masked_softmax_backward_non_contiguous():
    torch.manual_seed(0)
    x_base = torch.randn(64, 32, dtype=torch.float32)
    g_base = torch.randn(64, 32, dtype=torch.float32)
    m_base = torch.rand(64, 32) < 0.5
    x, g, m = x_base.t(), g_base.t(), m_base.t()  # (32, 64) views
    assert not x.is_contiguous() and not g.is_contiguous() and not m.is_contiguous()
    output = _forward_ref(x, m, -1)

    grad_input = _masked_softmax_backward(g, output, m, -1)

    assert grad_input.shape == output.shape
    torch.testing.assert_close(
        grad_input, _grad_ref(x.contiguous(), m.contiguous(), -1, g.contiguous()), rtol=1e-4, atol=1e-4
    )


def test_masked_softmax_backward_inputs_not_modified():
    torch.manual_seed(0)
    grad_output = torch.randn(16, 64, dtype=torch.float32)
    output = torch.randn(16, 64, dtype=torch.float32).softmax(-1)
    mask = torch.rand(16, 64) < 0.5
    g_snap, o_snap, m_snap = grad_output.clone(), output.clone(), mask.clone()

    _masked_softmax_backward(grad_output, output, mask, -1)

    assert torch.equal(grad_output, g_snap), "op must not touch grad_output"
    assert torch.equal(output, o_snap), "op must not touch output"
    assert torch.equal(mask, m_snap), "op must not touch mask"


def test_masked_softmax_backward_error_dim_out_of_range():
    grad_output = torch.randn(8, 64, dtype=torch.float32)
    output = torch.randn(8, 64, dtype=torch.float32).softmax(-1)
    mask = torch.rand(8, 64) < 0.5
    with pytest.raises(AssertionError, match="Invalid dim for masked softmax backward"):
        _masked_softmax_backward(grad_output, output, mask, 9)


def test_masked_softmax_backward_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel.
    mod = _spy_module()

    launches = []
    real_run = mod._masked_softmax_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._masked_softmax_backward_kernel, "run", spy_run)
    x = torch.randn(8, 64, dtype=torch.float32)
    mask = torch.rand(8, 64) < 0.5
    grad_output = torch.randn(8, 64, dtype=torch.float32)
    output = _forward_ref(x, mask, -1)
    grad_input = mod._masked_softmax_backward(grad_output, output, mask, -1)

    assert len(launches) == 1, "_masked_softmax_backward must use the Triton kernel"
    assert launches[0] == (8,)  # one program per reduction row (outer*inner)
    torch.testing.assert_close(
        grad_input, _grad_ref(x, mask, -1, grad_output), rtol=1e-4, atol=1e-4
    )
