import importlib

import pytest
import torch

from .mixed_dtypes_linear import mixed_dtypes_linear

# fp16 input/output with fp32 accumulation: the epilogue cast rounds at
# ~1e-3 relative, so atol 1e-2 covers the dtype floor with margin.

# oracle: formula per upstream mixed_dtypes_linear.py (aten op has no CPU
# kernel), mirroring the upstream test reference: int8 weights are biased
# uint8 (w - 128), int4 weights are biased nibbles ((q & 0xF) - 8, low
# nibble = even output column), scale applies per output channel and the
# activation (none/relu/silu) applies after the optional bias. fp16 computes
# the epilogue scale after an fp32 dot; bf16 scales the weight tiles in-loop
# (kernel lines 111-129) — same math, different rounding.


def _reference(input_tensor, weight, scale, mode, bias, activation):
    K, ncols = weight.shape
    N = scale.shape[0]
    if mode == "int8":
        w = weight.to(torch.int32) - 128
    else:
        lo = (weight.to(torch.int32) & 0xF) - 8
        hi = ((weight.to(torch.int32) >> 4) & 0xF) - 8
        w = torch.stack([lo, hi], dim=-1).reshape(K, N)

    if input_tensor.dtype == torch.float16:
        wf = w.to(input_tensor.dtype)
        acc = torch.matmul(input_tensor.float(), wf.float())
        acc = acc * scale.float()[None, :]
    else:
        wdq = (w.float() * scale.float()[None, :]).to(input_tensor.dtype)
        acc = torch.matmul(input_tensor.float(), wdq.float())

    if bias is not None:
        acc = acc + bias.float()[None, :]
    if activation == "relu":
        acc = torch.relu(acc)
    elif activation == "silu":
        acc = acc * torch.sigmoid(acc)
    return acc.to(input_tensor.dtype)


@pytest.mark.parametrize(
    "M, K, N, mode, has_bias, activation",
    [
        (16, 64, 32, "int8", True, "none"),
        (8, 64, 24, "int8", False, "relu"),
        (16, 64, 32, "int8", True, "silu"),
        (16, 64, 64, "int4", True, "none"),
        (8, 128, 64, "int4", False, "relu"),
        (16, 128, 32, "int4", True, "silu"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_dtypes_linear_matches_formula(
    M, K, N, mode, has_bias, activation, dtype
):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype) * 0.25
    ncols = N if mode == "int8" else N // 2
    weight = torch.randint(0, 256, (K, ncols), dtype=torch.uint8)
    scale = torch.randn(N, dtype=dtype) * 0.05
    bias = torch.randn(N, dtype=dtype) * 0.1 if has_bias else None

    out = mixed_dtypes_linear(x, weight, scale, bias=bias, activation=activation)
    ref = _reference(x, weight, scale, mode, bias, activation)
    assert out.shape == (M, N) and out.dtype == dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_mixed_dtypes_linear_3d_input():
    torch.manual_seed(1)
    dtype = torch.float16
    x = torch.randn(2, 3, 64, dtype=dtype) * 0.25
    weight = torch.randint(0, 256, (64, 32), dtype=torch.uint8)
    scale = torch.randn(32, dtype=dtype) * 0.05
    out = mixed_dtypes_linear(x, weight, scale)
    ref = _reference(x.reshape(-1, 64), weight, scale, "int8", None, "none").reshape(
        2, 3, 32
    )
    assert out.shape == (2, 3, 32)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_mixed_dtypes_linear_relu_clamps_negatives():
    torch.manual_seed(2)
    dtype = torch.float16
    x = torch.randn(8, 64, dtype=dtype) * 0.25
    weight = torch.randint(0, 256, (64, 16), dtype=torch.uint8)
    scale = (-torch.ones(16, dtype=dtype) * 0.05).abs()
    bias = -torch.ones(16, dtype=dtype) * 100.0  # push everything negative
    # (random uint8 weights give |acc| up to ~12 for this shape, so the bias
    # must dominate that spread for the all-clamped pin to be satisfiable)
    out = mixed_dtypes_linear(x, weight, scale, bias=bias, activation="relu")
    assert torch.all(out == 0)


def test_mixed_dtypes_linear_errors_are_assertions():
    torch.manual_seed(3)
    x = torch.randn(8, 64, dtype=torch.float16)
    weight = torch.randint(0, 256, (64, 32), dtype=torch.uint8)
    scale = torch.randn(32, dtype=torch.float16) * 0.05
    with pytest.raises(AssertionError):
        mixed_dtypes_linear(x.float(), weight, scale.half())  # fp32 input rejected
    with pytest.raises(AssertionError):
        mixed_dtypes_linear(x, weight.int(), scale)  # weight must be uint8
    with pytest.raises(AssertionError):
        mixed_dtypes_linear(x, weight, scale.float())  # scale dtype must match
    with pytest.raises(AssertionError):
        mixed_dtypes_linear(x, weight, scale, activation="gelu")  # unknown act
    with pytest.raises(AssertionError):
        mixed_dtypes_linear(
            x, torch.randint(0, 256, (64, 31), dtype=torch.uint8), scale
        )


def test_mixed_dtypes_linear_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.mixed_dtypes_linear")
    launches = []
    kernel = mod._mixed_dtypes_linear_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(16, 64, dtype=torch.float16) * 0.25
    weight = torch.randint(0, 256, (64, 32), dtype=torch.uint8)
    scale = torch.randn(32, dtype=torch.float16) * 0.05
    out = mixed_dtypes_linear(x, weight, scale)
    ref = _reference(x, weight, scale, "int8", None, "none")
    assert len(launches) >= 1
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
