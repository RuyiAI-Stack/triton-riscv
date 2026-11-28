import importlib

import pytest
import torch

from ._scaled_mm_v2 import _scaled_mm_v2

# Recipe codes from upstream _scaled_mm_v2.py (RECIPE_TENSOR_WISE / RECIPE_ROW_WISE).
TENSOR_WISE = 0
ROW_WISE = 1


def _fp8_a(shape):
    return (torch.randn(shape, dtype=torch.float32) * 0.25).to(torch.float8_e4m3fn)


def _mat_b_colmajor(k, n):
    # The kernel contracts along mat_b's leading dimension and requires
    # stride(0) == 1 (column major), so materialize (N, K) then swap.
    return (torch.randn((n, k), dtype=torch.float32) * 0.25).to(torch.float8_e4m3fn).t()


def _aten_v2(mat_a, mat_b, scale_a, recipe_a, scale_b, recipe_b, out_dtype):
    # aten::_scaled_mm_v2 runs on CPU when the scales are 2-D/(M,1)/(1,N)
    # tensors (probed on torch 2.15.0+cpu): direct oracle, same call shape.
    return torch.ops.aten._scaled_mm_v2(
        mat_a,
        mat_b,
        [scale_a],
        [recipe_a],
        [0],
        [scale_b],
        [recipe_b],
        [0],
        None,
        out_dtype,
        [],
        False,
    )


@pytest.mark.parametrize("M, N, K", [(32, 48, 64), (64, 96, 128), (17, 32, 65)])
@pytest.mark.parametrize("recipe", [TENSOR_WISE, ROW_WISE])
def test_scaled_mm_v2_matches_aten(M, N, K, recipe):
    torch.manual_seed(0)
    mat_a = _fp8_a((M, K))
    mat_b = _mat_b_colmajor(K, N)
    if recipe == TENSOR_WISE:
        scale_a = torch.tensor(0.9)
        scale_b = torch.tensor(1.1)
    else:
        scale_a = torch.linspace(0.75, 1.25, M).reshape(M, 1)
        scale_b = torch.linspace(1.25, 0.75, N).reshape(1, N)

    out = _scaled_mm_v2(
        mat_a,
        mat_b,
        [scale_a],
        [recipe],
        [0],
        [scale_b],
        [recipe],
        [0],
        None,
        torch.float32,
    )
    ref = _aten_v2(mat_a, mat_b, scale_a, recipe, scale_b, recipe, torch.float32)
    assert out.shape == (M, N) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_scaled_mm_v2_default_out_dtype_is_input_dtype():
    # Upstream _resolve_out_dtype: out_dtype None -> mat_a.dtype.
    torch.manual_seed(1)
    mat_a = _fp8_a((16, 32))
    mat_b = _mat_b_colmajor(32, 24)
    out = _scaled_mm_v2(
        mat_a,
        mat_b,
        [torch.rand(16)],
        [ROW_WISE],
        [0],
        [torch.rand(24)],
        [ROW_WISE],
        [0],
    )
    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == (16, 24)


def test_scaled_mm_v2_rowwise_1d_scales():
    # _prepare_scale flattens 1-D RowWise scales to length M / N.
    torch.manual_seed(2)
    M, N, K = 24, 32, 64
    mat_a = _fp8_a((M, K))
    mat_b = _mat_b_colmajor(K, N)
    scale_a = torch.linspace(0.75, 1.25, M)
    scale_b = torch.linspace(1.25, 0.75, N)

    out = _scaled_mm_v2(
        mat_a,
        mat_b,
        [scale_a],
        [ROW_WISE],
        [0],
        [scale_b],
        [ROW_WISE],
        [0],
        None,
        torch.float32,
    )
    ref = _aten_v2(
        mat_a,
        mat_b,
        scale_a.reshape(M, 1),
        ROW_WISE,
        scale_b.reshape(1, N),
        ROW_WISE,
        torch.float32,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_scaled_mm_v2_bias_added_before_cast():
    # oracle: formula per upstream _scaled_mm_v2.py (kernel lines 167-169:
    # bias loaded to fp32 and added to the scaled accumulator; aten's CPU
    # _scaled_mm_v2 bias form is not pinned here).
    torch.manual_seed(3)
    M, N, K = 24, 32, 64
    mat_a = _fp8_a((M, K))
    mat_b = _mat_b_colmajor(K, N)
    scale_a = torch.linspace(0.75, 1.25, M)
    scale_b = torch.linspace(1.25, 0.75, N)
    bias = torch.randn(N)

    out = _scaled_mm_v2(
        mat_a,
        mat_b,
        [scale_a],
        [ROW_WISE],
        [0],
        [scale_b],
        [ROW_WISE],
        [0],
        bias,
        torch.float32,
    )
    ref = (mat_a.float() @ mat_b.float()) * scale_a.reshape(-1, 1) * scale_b.reshape(
        1, -1
    ) + bias.reshape(1, -1)
    torch.testing.assert_close(out, ref.to(torch.float32), rtol=1e-4, atol=1e-4)


def test_scaled_mm_v2_zero_scale_yields_zeros():
    torch.manual_seed(4)
    mat_a = _fp8_a((16, 32))
    mat_b = _mat_b_colmajor(32, 24)
    out = _scaled_mm_v2(
        mat_a,
        mat_b,
        [torch.tensor(0.0)],
        [TENSOR_WISE],
        [0],
        [torch.tensor(2.0)],
        [TENSOR_WISE],
        [0],
        None,
        torch.float32,
    )
    assert torch.all(out == 0)


def test_scaled_mm_v2_empty_rows():
    torch.manual_seed(5)
    mat_a = _fp8_a((0, 32))
    mat_b = _mat_b_colmajor(32, 24)
    out = _scaled_mm_v2(
        mat_a,
        mat_b,
        [torch.rand(0)],
        [ROW_WISE],
        [0],
        [torch.rand(24)],
        [ROW_WISE],
        [0],
        None,
        torch.float32,
    )
    assert out.shape == (0, 24)


def test_scaled_mm_v2_errors_match_upstream_messages():
    torch.manual_seed(6)
    mat_a = _fp8_a((16, 32))
    mat_b = _mat_b_colmajor(32, 24)
    row = ([torch.rand(16)], [ROW_WISE], [0], [torch.rand(24)], [ROW_WISE], [0])

    with pytest.raises(ValueError, match="Expected mat2 to be transposed"):
        _scaled_mm_v2(mat_a, mat_b.t().contiguous(), *row, None, torch.float32)
    with pytest.raises(ValueError, match="_scaled_mm_v2 expects 2D operands"):
        _scaled_mm_v2(mat_a.unsqueeze(0), mat_b, *row, None, torch.float32)
    with pytest.raises(ValueError, match="must have the same dtype"):
        _scaled_mm_v2(
            mat_a,
            mat_b.to(torch.float8_e5m2),
            *row,
            None,
            torch.float32,
        )
    with pytest.raises(ValueError, match="expected mat_a to be a float8 matrix"):
        _scaled_mm_v2(mat_a.float(), mat_b.float(), *row, None, torch.float32)
    with pytest.raises(
        ValueError, match="contraction_dim is only supported for batched operands"
    ):
        _scaled_mm_v2(mat_a, mat_b, *row, None, torch.float32, [1], False)
    with pytest.raises(
        ValueError, match="recipe_a must describe exactly one scaling recipe"
    ):
        _scaled_mm_v2(
            mat_a, mat_b, [torch.rand(16)], [ROW_WISE, ROW_WISE], [0], *row[3:]
        )
    with pytest.raises(ValueError, match="unsupported scaling recipe"):
        _scaled_mm_v2(mat_a, mat_b, [torch.rand(16)], [7], [0], *row[3:])
    with pytest.raises(
        ValueError, match="swizzle_a must describe exactly one swizzle pattern"
    ):
        _scaled_mm_v2(mat_a, mat_b, [torch.rand(16)], [ROW_WISE], [0, 0], *row[3:])
    with pytest.raises(ValueError, match="unsupported swizzle pattern"):
        _scaled_mm_v2(mat_a, mat_b, [torch.rand(16)], [ROW_WISE], [1], *row[3:])
    with pytest.raises(ValueError, match="scale_a must contain exactly one tensor"):
        _scaled_mm_v2(
            mat_a, mat_b, [torch.rand(16), torch.rand(16)], [ROW_WISE], [0], *row[3:]
        )
    with pytest.raises(ValueError, match="scale_a must be a list of tensors"):
        _scaled_mm_v2(mat_a, mat_b, None, [ROW_WISE], [0], *row[3:])
    with pytest.raises(ValueError, match="scale_a must be float32"):
        _scaled_mm_v2(
            mat_a,
            mat_b,
            [torch.rand(16, dtype=torch.float64)],
            [ROW_WISE],
            [0],
            *row[3:],
        )
    with pytest.raises(ValueError, match="scale_a must hold one element"):
        _scaled_mm_v2(mat_a, mat_b, [torch.rand(3)], [TENSOR_WISE], [0], *row[3:])
    with pytest.raises(ValueError, match="scale_a must have the same length as mat"):
        _scaled_mm_v2(mat_a, mat_b, [torch.rand(3)], [ROW_WISE], [0], *row[3:])
    with pytest.raises(ValueError, match="bias must have"):
        _scaled_mm_v2(mat_a, mat_b, *row, torch.zeros(5), torch.float32)
    with pytest.raises(ValueError, match="Unsupported out_dtype for _scaled_mm_v2"):
        _scaled_mm_v2(mat_a, mat_b, *row, None, torch.int64)


def test_scaled_mm_v2_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._scaled_mm_v2")
    launches = []
    kernel = mod._scaled_mm_v2_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    mat_a = _fp8_a((32, 64))
    mat_b = _mat_b_colmajor(64, 48)
    scale_a = torch.linspace(0.75, 1.25, 32).reshape(32, 1)
    scale_b = torch.linspace(1.25, 0.75, 48).reshape(1, 48)
    out = _scaled_mm_v2(
        mat_a,
        mat_b,
        [scale_a],
        [ROW_WISE],
        [0],
        [scale_b],
        [ROW_WISE],
        [0],
        None,
        torch.float32,
    )
    ref = _aten_v2(mat_a, mat_b, scale_a, ROW_WISE, scale_b, ROW_WISE, torch.float32)
    assert len(launches) >= 1
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
