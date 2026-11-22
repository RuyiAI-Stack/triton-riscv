import importlib

import pytest
import torch

from .scaled_grouped_mm import scaled_grouped_mm

# Tolerances follow the upstream suite budget for grouped scaled matmul:
# upstream tests/test_scaled_grouped_mm*.py compare against the fp32
# reference with atol=2.5e-1, rtol=5e-1 for bf16/fp8 outputs.

# oracle: formula per upstream scaled_grouped_mm.py (no aten CPU op) — the
# per-group composition mirrors _scaled_grouped_mm_fallback / the upstream
# test reference: chunk = mat_a_slice.float() @ mat_b_slice.float(), then
# out = chunk * scale_a_row * scale_b_col (+ bias), cast to out_dtype.


def _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, bias, out_dtype):
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    if a_is_2d and b_is_2d:
        groups = offs.numel()
    elif a_is_2d:
        groups = mat_b.shape[0]
    else:
        groups = mat_a.shape[0]

    mat_a = mat_a.float()
    mat_b = mat_b.float()
    scale_a = scale_a.float()
    scale_b = scale_b.float()
    offsets = [0] + (offs.tolist() if offs is not None else [])

    def scale_and_bias(out, sa, sb, chunk_bias):
        out = out * sa * sb
        if chunk_bias is not None:
            out = out + chunk_bias
        return out.to(out_dtype)

    chunks = []
    if a_is_2d and not b_is_2d:
        for g in range(groups):
            m0, m1 = offsets[g], offsets[g + 1]
            chunk = mat_a[m0:m1].mm(mat_b[g])
            cb = bias if bias is None or bias.dim() == 1 else bias[g]
            chunks.append(
                scale_and_bias(
                    chunk, scale_a[m0:m1].reshape(-1, 1), scale_b[g].reshape(1, -1), cb
                )
            )
        return torch.cat(chunks, dim=0)

    if not a_is_2d and b_is_2d:
        for g in range(groups):
            n0, n1 = offsets[g], offsets[g + 1]
            chunk = mat_a[g].mm(mat_b[:, n0:n1])
            cb = bias[n0:n1] if bias is not None else None
            chunks.append(
                scale_and_bias(
                    chunk, scale_a[g].reshape(-1, 1), scale_b[n0:n1].reshape(1, -1), cb
                )
            )
        return torch.cat(chunks, dim=1)

    if a_is_2d and b_is_2d:
        scale_a = scale_a.reshape(groups, mat_a.shape[0])
        scale_b = scale_b.reshape(groups, mat_b.shape[1])
        for g in range(groups):
            k0, k1 = offsets[g], offsets[g + 1]
            chunk = mat_a[:, k0:k1].mm(mat_b[k0:k1])
            cb = bias if bias is None or bias.dim() == 1 else bias[g]
            chunks.append(
                scale_and_bias(
                    chunk, scale_a[g].reshape(-1, 1), scale_b[g].reshape(1, -1), cb
                )
            )
        return torch.stack(chunks, dim=0)

    for g in range(groups):
        chunk = mat_a[g].mm(mat_b[g])
        cb = bias if bias is None or bias.dim() == 1 else bias[g]
        chunks.append(
            scale_and_bias(
                chunk, scale_a[g].reshape(-1, 1), scale_b[g].reshape(1, -1), cb
            )
        )
    return torch.stack(chunks, dim=0)


def _case(case, dtype):
    groups, M, N, K = 3, 5, 16, 32
    make = lambda shape: (torch.randn(shape) * 0.25).to(dtype)  # noqa: E731

    if case == "m_varying":
        offs = torch.tensor([2, 5, 9], dtype=torch.int32)
        return make((9, K)), make((groups, K, N)), offs
    if case == "n_varying":
        offs = torch.tensor([8, 20, 32], dtype=torch.int32)
        return make((groups, M, K)), make((K, 32)), offs
    if case == "k_varying":
        offs = torch.tensor([8, 20, K], dtype=torch.int32)
        return make((M, K)), make((K, N)), offs
    return make((groups, M, K)), make((groups, K, N)), None


def _scales(case, dtype, mat_a, mat_b, offs):
    groups = offs.numel() if offs is not None else mat_a.shape[0]
    if case == "m_varying":
        return (
            torch.linspace(0.75, 1.25, mat_a.shape[0]),
            torch.linspace(1.25, 0.75, groups * mat_b.shape[-1]).reshape(groups, -1),
        )
    if case == "n_varying":
        return (
            torch.linspace(0.75, 1.25, groups * mat_a.shape[1]).reshape(groups, -1),
            torch.linspace(1.25, 0.75, mat_b.shape[1]),
        )
    if case == "k_varying":
        return (
            torch.linspace(0.75, 1.25, groups * mat_a.shape[0]),
            torch.linspace(1.25, 0.75, groups * mat_b.shape[1]),
        )
    return (
        torch.linspace(0.75, 1.25, groups * mat_a.shape[1]).reshape(groups, -1),
        torch.linspace(1.25, 0.75, groups * mat_b.shape[-1]).reshape(groups, -1),
    )


@pytest.mark.parametrize("case", ["m_varying", "n_varying", "k_varying", "batch"])
def test_scaled_grouped_mm_float32_matches_formula(case):
    torch.manual_seed(0)
    dtype = torch.float32
    mat_a, mat_b, offs = _case(case, dtype)
    scale_a, scale_b = _scales(case, dtype, mat_a, mat_b, offs)

    ref = _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, None, torch.float32)
    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs)
    assert out.shape == ref.shape and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_scaled_grouped_mm_float16_default_out_dtype():
    torch.manual_seed(1)
    mat_a, mat_b, offs = _case("m_varying", torch.float16)
    scale_a, scale_b = _scales("m_varying", torch.float16, mat_a, mat_b, offs)

    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs)
    assert out.dtype == torch.float16

    ref = _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, None, torch.float16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_scaled_grouped_mm_fp8_e4m3_default_out_dtype_bf16():
    torch.manual_seed(2)
    mat_a, mat_b, offs = _case("m_varying", torch.float8_e4m3fn)
    scale_a, scale_b = _scales("m_varying", torch.float8_e4m3fn, mat_a, mat_b, offs)

    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs)
    assert out.dtype == torch.bfloat16

    ref = _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, None, torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-1, rtol=5e-1)


def test_scaled_grouped_mm_non_uniform_groups_two_batches():
    # Two different group partitions must both land on the formula oracle.
    torch.manual_seed(3)
    for sizes in ([1, 8], [5, 4], [7, 2]):
        offs = torch.tensor([sizes[0], sizes[0] + sizes[1]], dtype=torch.int32)
        mat_a = (torch.randn((sum(sizes), 32)) * 0.25).to(torch.float32)
        mat_b = (torch.randn((2, 32, 16)) * 0.25).to(torch.float32)
        scale_a = torch.linspace(0.75, 1.25, sum(sizes))
        scale_b = torch.linspace(1.25, 0.75, 2 * 16).reshape(2, 16)
        out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs)
        ref = _grouped_reference(
            mat_a, mat_b, scale_a, scale_b, offs, None, torch.float32
        )
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4, msg=f"sizes={sizes}")


def test_scaled_grouped_mm_single_group():
    torch.manual_seed(4)
    mat_a = (torch.randn((4, 32)) * 0.25).to(torch.float32)
    mat_b = (torch.randn((1, 32, 16)) * 0.25).to(torch.float32)
    offs = torch.tensor([4], dtype=torch.int32)
    scale_a = torch.linspace(0.75, 1.25, 4)
    scale_b = torch.linspace(1.25, 0.75, 16).reshape(1, 16)
    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs)
    ref = _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, None, torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("bias_kind", ["vector", "grouped"])
@pytest.mark.parametrize("case", ["m_varying", "batch"])
def test_scaled_grouped_mm_bias(bias_kind, case):
    torch.manual_seed(5)
    mat_a, mat_b, offs = _case(case, torch.float32)
    scale_a, scale_b = _scales(case, torch.float32, mat_a, mat_b, offs)
    groups = offs.numel() if offs is not None else mat_a.shape[0]
    n = mat_b.shape[-1]
    bias = torch.randn(n) if bias_kind == "vector" else torch.randn(groups, n)

    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs, bias=bias)
    ref = _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, bias, torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_scaled_grouped_mm_out_dtype_override():
    torch.manual_seed(6)
    mat_a, mat_b, offs = _case("batch", torch.float16)
    scale_a, scale_b = _scales("batch", torch.float16, mat_a, mat_b, offs)
    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, out_dtype=torch.float32)
    assert out.dtype == torch.float32
    ref = _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, None, torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_scaled_grouped_mm_fp8_identity_exact():
    # Every finite e4m3 encoding times the identity must reproduce the decoded
    # values exactly (upstream test_scaled_grouped_mm_fp8_encodings pin).
    bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    bits[(bits & 127) == 127] = 0  # drop the NaN encodings
    a = bits.view(torch.float8_e4m3fn).reshape(1, 16, 16)
    b = torch.eye(16).to(torch.float8_e4m3fn).reshape(1, 16, 16)
    scales = torch.ones((1, 16))
    out = scaled_grouped_mm(a, b, scales, scales, out_dtype=torch.float32)
    torch.testing.assert_close(out, a.float(), rtol=0, atol=0)


def test_scaled_grouped_mm_scales_before_output_cast():
    # Unscaled product 64*64*16 = 65536 overflows fp16; scaling first keeps
    # 16384 exact (upstream test pin).
    mat_a = torch.full((1, 3, 16), 64.0, dtype=torch.float16)
    mat_b = torch.full((1, 16, 5), 64.0, dtype=torch.float16)
    scale_a = torch.full((1, 3), 0.5)
    scale_b = torch.full((1, 5), 0.5)
    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b)
    expected = torch.full((1, 3, 5), 16384.0, dtype=torch.float16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_scaled_grouped_mm_zero_scale():
    torch.manual_seed(7)
    mat_a, mat_b, offs = _case("m_varying", torch.float32)
    scale_a = torch.zeros(9)
    scale_b = torch.linspace(1.25, 0.75, 3 * 16).reshape(3, 16)
    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs)
    assert torch.all(out == 0)


def test_scaled_grouped_mm_non_contiguous_mat_a():
    # Non-contiguous rows on the M axis (stride > 1 in both dims triggers the
    # wrapper's contiguous() fallback in the upstream implementation).
    torch.manual_seed(8)
    base = torch.randn(18, 64)
    mat_a = base[::2]  # (9, 64), stride (128, 1) -> non-contiguous
    assert not mat_a.is_contiguous()
    mat_b = (torch.randn((3, 64, 16)) * 0.25).to(torch.float32)
    offs = torch.tensor([2, 5, 9], dtype=torch.int32)
    scale_a = torch.linspace(0.75, 1.25, 9)
    scale_b = torch.linspace(1.25, 0.75, 3 * 16).reshape(3, 16)
    out = scaled_grouped_mm(mat_a.contiguous(), mat_b, scale_a, scale_b, offs=offs)
    ref = _grouped_reference(
        mat_a.contiguous(), mat_b, scale_a, scale_b, offs, None, torch.float32
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_scaled_grouped_mm_errors_match_upstream_messages():
    torch.manual_seed(9)
    mat_a, mat_b, offs = _case("m_varying", torch.float32)
    scale_a, scale_b = _scales("m_varying", torch.float32, mat_a, mat_b, offs)
    with pytest.raises(RuntimeError, match="scale_result is not supported"):
        scaled_grouped_mm(
            mat_a, mat_b, scale_a, scale_b, offs=offs, scale_result=torch.tensor(1.0)
        )
    with pytest.raises(RuntimeError, match="offs has to be int32"):
        scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs.to(torch.int64))
    with pytest.raises(
        RuntimeError, match="offs must be provided when either input is 2D"
    ):
        scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=None)
    with pytest.raises(RuntimeError, match="offs must be None when both inputs are 3D"):
        scaled_grouped_mm(
            *(torch.randn(3, 5, 32), torch.randn(3, 32, 16)),
            torch.linspace(0.75, 1.25, 15).reshape(3, 5),
            torch.linspace(1.25, 0.75, 48).reshape(3, 16),
            offs=offs,
        )
    with pytest.raises(RuntimeError, match="mat_a has to be 2D or 3D"):
        scaled_grouped_mm(
            torch.randn(32),
            mat_b,
            torch.linspace(0.75, 1.25, 9),
            scale_b,
            offs=offs,
        )
    with pytest.raises(RuntimeError, match="mat_a and mat_b must have the same dtype"):
        scaled_grouped_mm(
            mat_a,
            mat_b.to(torch.float64),
            scale_a,
            scale_b,
            offs=offs,
        )
    with pytest.raises(
        RuntimeError, match="mat_a and mat_b shapes cannot be multiplied"
    ):
        scaled_grouped_mm(
            mat_a,
            mat_b[:, :8, :],
            scale_a,
            torch.linspace(1.25, 0.75, 3 * 8).reshape(3, 8),
            offs=offs,
        )
    with pytest.raises(RuntimeError, match="bias must have shape"):
        scaled_grouped_mm(
            mat_a, mat_b, scale_a, scale_b, offs=offs, bias=torch.zeros(5)
        )


def test_scaled_grouped_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.scaled_grouped_mm")
    launches = []
    kernel = mod.scaled_grouped_mm_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    mat_a, mat_b, offs = _case("m_varying", torch.float32)
    scale_a, scale_b = _scales("m_varying", torch.float32, mat_a, mat_b, offs)
    out = scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs=offs)
    ref = _grouped_reference(mat_a, mat_b, scale_a, scale_b, offs, None, torch.float32)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
