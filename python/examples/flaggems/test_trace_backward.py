import importlib

import pytest
import torch

from .trace_backward import trace_backward

# Reference (probed against torch 2.15 CPU): aten::trace_backward(g, [N, M])
# returns a zero (N, M) tensor of grad's dtype with g on the main diagonal
# -- min(N, M) entries, i.e. grad * eye(N, M).  Errors: sizes of length
# != 2 -> RuntimeError("expected matrix input"); a non-0-dim grad ->
# RuntimeError from index_fill_; negative sizes -> the zeros() error.
# Upstream FlagGems counts ceil(N * M / (M + 1)) diagonal slots, which
# overshoots for skinny matrices ((5, 1) would fill three entries where
# aten fills one) -- pinned to aten here.


@pytest.mark.parametrize(
    "sizes",
    [
        (4, 4),
        (1, 1),
        (2, 3),
        (3, 2),
        (7, 2),
        (2, 7),
        (5, 2),
        (2, 5),
        (6, 3),
        (3, 6),
        (5, 1),
        (1, 5),
        (2049, 1),
        (1, 2049),
        (1500, 1200),  # diagonal longer than one block
        (1200, 1500),
    ],
)
@pytest.mark.parametrize("grad_value", [3.0, -2.5, 0.0])
def test_trace_backward_vs_aten(sizes, grad_value):
    grad = torch.tensor(grad_value)

    ref = torch.ops.aten.trace_backward(grad, list(sizes))
    out = trace_backward(grad, sizes)

    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("sizes", [(4, 4), (3, 5), (5, 3), (5, 1), (1, 5)])
def test_trace_backward_eye_semantics(sizes):
    # The gradient of a matrix trace is grad * I, including the
    # rectangular identity for non-square and skinny inputs.
    grad = torch.tensor(2.5)
    out = trace_backward(grad, sizes)

    ref = grad * torch.eye(*sizes)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert (out != 0).sum().item() == min(sizes)


@pytest.mark.parametrize("sizes", [(3, 5), (5, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_trace_backward_autograd_crosscheck(sizes, dtype):
    # End-to-end: autograd through torch.trace must produce exactly the
    # op's output for non-square inputs.
    a = torch.randn(sizes, dtype=dtype, requires_grad=True)
    upstream = torch.tensor(1.25, dtype=dtype)
    a.trace().backward(upstream)

    out = trace_backward(upstream, sizes)

    torch.testing.assert_close(out, a.grad, rtol=0, atol=0)


def test_trace_backward_preserves_dtype():
    # grad dtype flows through unchanged (int grads stay int).
    for dtype in (torch.float32, torch.float64, torch.float16, torch.int64):
        grad = torch.tensor(3, dtype=dtype)
        out = trace_backward(grad, (2, 2))
        assert out.dtype == dtype
        torch.testing.assert_close(
            out, torch.ops.aten.trace_backward(grad, [2, 2]), rtol=0, atol=0
        )


@pytest.mark.parametrize("sizes", [(0, 0), (0, 5), (5, 0)])
def test_trace_backward_empty(sizes):
    grad = torch.tensor(3.0)
    out = trace_backward(grad, sizes)

    assert out.shape == torch.Size(sizes)
    assert out.numel() == 0
    torch.testing.assert_close(
        out, torch.ops.aten.trace_backward(grad, list(sizes)), rtol=0, atol=0
    )


def test_trace_backward_rejects_non_matrix_sizes():
    grad = torch.tensor(3.0)
    with pytest.raises(RuntimeError, match="expected matrix input"):
        trace_backward(grad, [2, 3, 4])
    with pytest.raises(RuntimeError, match="expected matrix input"):
        trace_backward(grad, [2])


def test_trace_backward_rejects_non_scalar_grad():
    grad = torch.tensor([1.0, 2.0])
    with pytest.raises(
        RuntimeError, match="index_fill_ only supports a 0-dimensional value tensor"
    ):
        trace_backward(grad, [2, 2])


def test_trace_backward_rejects_negative_sizes():
    grad = torch.tensor(3.0)
    with pytest.raises(RuntimeError, match="Dimension size must be non-negative"):
        trace_backward(grad, [-2, 3])


def test_trace_backward_triton_path(monkeypatch):
    # Smoke: the main path must launch the Triton kernel.
    mod = importlib.import_module("flaggems.trace_backward")

    launches = []
    real_run = mod.trace_backward_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod.trace_backward_kernel, "run", spy_run)
    out = mod.trace_backward(torch.tensor(3.0), (4, 4))
    assert len(launches) == 1, "trace_backward must go through the Triton kernel"
    assert launches[0] == (1,)
    assert out[0, 0].item() == 3.0 and out[1, 1].item() == 3.0
