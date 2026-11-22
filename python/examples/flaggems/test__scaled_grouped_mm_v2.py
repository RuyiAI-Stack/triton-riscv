import importlib

import pytest
import torch

from ._scaled_grouped_mm_v2 import _scaled_grouped_mm_v2

# Tolerances follow the upstream suite budget for grouped scaled matmul:
# upstream tests/test_scaled_grouped_mm_v2.py compares against the fp32
# reference with atol=2.5e-1, rtol=5e-1 for bf16/fp8 outputs.

# oracle: formula per upstream _scaled_grouped_mm_v2.py (aten op has no CPU
# kernel): each group contributes mat_a_slice.float() @ mat_b_slice.float()
# scaled by the per-row scale_a / per-column scale_b, cast to bf16 (the only
# supported high-precision output dtype).

ROW_WISE = 1
TENSOR_WISE = 0


def _fp8(shape):
    return (torch.randn(shape) * 0.25).to(torch.float8_e4m3fn)


def _mat_b_colmajor(shape):
    # mat_b must arrive transposed (stride(-2) == 1): materialize (..., N, K)
    # then swap the last two dims.
    *lead, k, n = shape
    return (torch.randn(lead + [n, k]) * 0.25).to(torch.float8_e4m3fn).transpose(-2, -1)


def _case(case):
    groups, m_per_group, n_per_group, k_size = 3, 32, 64, 128
    if case == "m_varying":
        offs = torch.tensor([16, 48, 96], dtype=torch.int32)
        mat_a = _fp8((96, k_size))
        mat_b = _mat_b_colmajor((groups, k_size, n_per_group))
        scale_a = torch.linspace(0.75, 1.25, 96)  # jagged: one per M row
        scale_b = torch.linspace(1.25, 0.75, groups * n_per_group).reshape(
            groups, n_per_group
        )
        return mat_a, mat_b, scale_a, scale_b, offs
    if case == "n_varying":
        offs = torch.tensor([16, 32, 64], dtype=torch.int32)
        mat_a = _fp8((groups, m_per_group, k_size))
        mat_b = _mat_b_colmajor((k_size, 64))
        scale_a = torch.linspace(0.75, 1.25, groups * m_per_group).reshape(
            groups, m_per_group
        )
        scale_b = torch.linspace(1.25, 0.75, 64)  # jagged: one per N column
        return mat_a, mat_b, scale_a, scale_b, offs
    if case == "k_varying":
        offs = torch.tensor([32, 64, 96], dtype=torch.int32)
        mat_a = _fp8((m_per_group, k_size))
        mat_b = _mat_b_colmajor((k_size, n_per_group))
        scale_a = torch.linspace(0.75, 1.25, groups * m_per_group)
        scale_b = torch.linspace(1.25, 0.75, groups * n_per_group)
        return mat_a, mat_b, scale_a, scale_b, offs
    mat_a = _fp8((groups, m_per_group, k_size))
    mat_b = _mat_b_colmajor((groups, k_size, n_per_group))
    scale_a = torch.linspace(0.75, 1.25, groups * m_per_group).reshape(
        groups, m_per_group
    )
    scale_b = torch.linspace(1.25, 0.75, groups * n_per_group).reshape(
        groups, n_per_group
    )
    return mat_a, mat_b, scale_a, scale_b, None


def _reference(mat_a, mat_b, scale_a, scale_b, offs):
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

    chunks = []
    if a_is_2d and not b_is_2d:
        for g in range(groups):
            m0, m1 = offsets[g], offsets[g + 1]
            chunk = mat_a[m0:m1].mm(mat_b[g])
            chunks.append(
                chunk * scale_a[m0:m1].reshape(-1, 1) * scale_b[g].reshape(1, -1)
            )
        return torch.cat(chunks, dim=0)
    if not a_is_2d and b_is_2d:
        for g in range(groups):
            n0, n1 = offsets[g], offsets[g + 1]
            chunk = mat_a[g].mm(mat_b[:, n0:n1])
            chunks.append(
                chunk * scale_a[g].reshape(-1, 1) * scale_b[n0:n1].reshape(1, -1)
            )
        return torch.cat(chunks, dim=1)
    if a_is_2d and b_is_2d:
        scale_a = scale_a.reshape(groups, mat_a.shape[0])
        scale_b = scale_b.reshape(groups, mat_b.shape[1])
        for g in range(groups):
            k0, k1 = offsets[g], offsets[g + 1]
            chunk = mat_a[:, k0:k1].mm(mat_b[k0:k1])
            chunks.append(chunk * scale_a[g].reshape(-1, 1) * scale_b[g].reshape(1, -1))
        return torch.stack(chunks, dim=0)
    for g in range(groups):
        chunk = mat_a[g].mm(mat_b[g])
        chunks.append(chunk * scale_a[g].reshape(-1, 1) * scale_b[g].reshape(1, -1))
    return torch.stack(chunks, dim=0)


def _run(mat_a, mat_b, scale_a, recipe_a, scale_b, recipe_b, offs, out_dtype):
    return _scaled_grouped_mm_v2(
        mat_a,
        mat_b,
        [scale_a],
        [recipe_a],
        [0],
        [scale_b],
        [recipe_b],
        [0],
        offs,
        None,
        out_dtype,
        [],
        False,
    )


@pytest.mark.parametrize("case", ["m_varying", "n_varying", "k_varying", "batch"])
def test_scaled_grouped_mm_v2_rowwise_matches_formula(case):
    torch.manual_seed(0)
    mat_a, mat_b, scale_a, scale_b, offs = _case(case)
    out = _run(mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, offs, torch.bfloat16)
    ref = _reference(mat_a, mat_b, scale_a, scale_b, offs).to(torch.bfloat16)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-1, rtol=5e-1)


def test_scaled_grouped_mm_v2_tensorwise():
    torch.manual_seed(1)
    mat_a, mat_b, _, _, offs = _case("m_varying")
    scale_a = torch.full((1,), 0.9)
    scale_b = torch.full((1,), 1.1)
    out = _run(
        mat_a, mat_b, scale_a, TENSOR_WISE, scale_b, TENSOR_WISE, offs, torch.bfloat16
    )

    groups = mat_b.shape[0]
    n = mat_b.shape[-1]
    ref = _reference(
        mat_a,
        mat_b,
        torch.full((mat_a.shape[0],), 0.9),
        torch.full((groups, n), 1.1),
        offs,
    ).to(torch.bfloat16)
    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-1, rtol=5e-1)


def test_scaled_grouped_mm_v2_default_out_dtype_is_bf16():
    torch.manual_seed(2)
    mat_a, mat_b, scale_a, scale_b, offs = _case("m_varying")
    out = _run(mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, offs, None)
    assert out.dtype == torch.bfloat16


def test_scaled_grouped_mm_v2_use_fast_accum_is_not_a_result_change():
    torch.manual_seed(3)
    mat_a, mat_b, scale_a, scale_b, offs = _case("m_varying")
    slow = _run(
        mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, offs, torch.bfloat16
    )
    fast = _scaled_grouped_mm_v2(
        mat_a,
        mat_b,
        [scale_a],
        [ROW_WISE],
        [0],
        [scale_b],
        [ROW_WISE],
        [0],
        offs,
        None,
        torch.bfloat16,
        [],
        True,
    )
    torch.testing.assert_close(slow.float(), fast.float(), atol=2.5e-1, rtol=5e-1)


def test_scaled_grouped_mm_v2_non_uniform_groups():
    torch.manual_seed(4)
    for sizes in ([16, 64, 16], [80, 8, 8]):
        offs = torch.tensor([sizes[0], sizes[0] + sizes[1], 96], dtype=torch.int32)
        mat_a = _fp8((96, 128))
        mat_b = _mat_b_colmajor((3, 128, 64))
        scale_a = torch.linspace(0.75, 1.25, 96)
        scale_b = torch.linspace(1.25, 0.75, 3 * 64).reshape(3, 64)
        out = _run(
            mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, offs, torch.bfloat16
        )
        ref = _reference(mat_a, mat_b, scale_a, scale_b, offs).to(torch.bfloat16)
        torch.testing.assert_close(
            out.float(), ref.float(), atol=2.5e-1, rtol=5e-1, msg=f"sizes={sizes}"
        )


def test_scaled_grouped_mm_v2_empty_output():
    torch.manual_seed(5)
    mat_a = _fp8((0, 128))
    mat_b = _mat_b_colmajor((3, 128, 64))
    offs = torch.tensor([0, 0, 0], dtype=torch.int32)
    out = _run(
        mat_a,
        mat_b,
        torch.zeros(0),
        ROW_WISE,
        torch.zeros(3, 64),
        ROW_WISE,
        offs,
        torch.bfloat16,
    )
    assert out.shape == (0, 64)


def test_scaled_grouped_mm_v2_errors_match_upstream_messages():
    torch.manual_seed(6)
    mat_a, mat_b, scale_a, scale_b, offs = _case("m_varying")
    with pytest.raises(ValueError, match="Bias not supported yet"):
        _scaled_grouped_mm_v2(
            mat_a,
            mat_b,
            [scale_a],
            [ROW_WISE],
            [0],
            [scale_b],
            [ROW_WISE],
            [0],
            offs,
            torch.zeros(64),
            torch.bfloat16,
            [],
            False,
        )
    with pytest.raises(ValueError, match="contraction_dim only applies"):
        _scaled_grouped_mm_v2(
            mat_a,
            mat_b,
            [scale_a],
            [ROW_WISE],
            [0],
            [scale_b],
            [ROW_WISE],
            [0],
            offs,
            None,
            torch.bfloat16,
            [1],
            False,
        )
    with pytest.raises(ValueError, match="Offsets have to be int32"):
        _run(
            mat_a,
            mat_b,
            scale_a,
            ROW_WISE,
            scale_b,
            ROW_WISE,
            offs.to(torch.int64),
            torch.bfloat16,
        )
    with pytest.raises(
        ValueError, match="Have to provide offsets if there is a 2d matrix"
    ):
        _run(mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, None, torch.bfloat16)
    with pytest.raises(ValueError, match="offs must be None when both operands are 3D"):
        _run(
            _fp8((3, 32, 128)),
            _mat_b_colmajor((3, 128, 64)),
            torch.rand(3, 32),
            ROW_WISE,
            torch.rand(3, 64),
            ROW_WISE,
            offs,
            torch.bfloat16,
        )
    with pytest.raises(ValueError, match="Expected mat2 to be transposed"):
        _run(
            mat_a,
            mat_b.transpose(-2, -1).contiguous(),
            scale_a,
            ROW_WISE,
            scale_b,
            ROW_WISE,
            offs,
            torch.bfloat16,
        )
    with pytest.raises(
        ValueError,
        match="Only bf16 high precision output types are supported for grouped gemm",
    ):
        _run(mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, offs, torch.float32)
    with pytest.raises(ValueError, match="expected mat_a to be a float8_e4m3 tensor"):
        _run(
            mat_a.float(),
            mat_b.float(),
            scale_a,
            ROW_WISE,
            scale_b,
            ROW_WISE,
            offs,
            torch.bfloat16,
        )
    with pytest.raises(ValueError, match="must have the same dtype"):
        _run(
            mat_a,
            mat_b.to(torch.float8_e5m2),
            scale_a,
            ROW_WISE,
            scale_b,
            ROW_WISE,
            offs,
            torch.bfloat16,
        )
    with pytest.raises(ValueError, match="matrix batch sizes have to match"):
        _run(
            _fp8((3, 32, 128)),
            _mat_b_colmajor((2, 128, 64)),
            torch.rand(3, 32),
            ROW_WISE,
            torch.rand(2, 64),
            ROW_WISE,
            None,
            torch.bfloat16,
        )
    with pytest.raises(
        ValueError, match="scale_a must be a 1D tensor for a jagged operand"
    ):
        _run(
            mat_a,
            mat_b,
            scale_a.reshape(3, 32),
            ROW_WISE,
            scale_b,
            ROW_WISE,
            offs,
            torch.bfloat16,
        )
    with pytest.raises(ValueError, match="scale_a must have the same length as mat"):
        _run(
            mat_a,
            mat_b,
            torch.rand(7),
            ROW_WISE,
            scale_b,
            ROW_WISE,
            offs,
            torch.bfloat16,
        )


def test_scaled_grouped_mm_v2_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._scaled_grouped_mm_v2")
    launches = []
    kernel = mod._scaled_grouped_mm_v2_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    mat_a, mat_b, scale_a, scale_b, offs = _case("m_varying")
    out = _run(mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, offs, torch.bfloat16)
    ref = _reference(mat_a, mat_b, scale_a, scale_b, offs).to(torch.bfloat16)
    assert len(launches) >= 1
    torch.testing.assert_close(out.float(), ref.float(), atol=2.5e-1, rtol=5e-1)
