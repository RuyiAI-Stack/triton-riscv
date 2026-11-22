import importlib

import pytest
import torch

from ._cummin_helper import _cummin_helper

# oracle: torch.cummin on CPU plus aten._cummin_helper (schema pinned on
# torch 2.15.0+cpu: (Tensor self, Tensor(a!) values, Tensor(b!) indices,
# int dim) -> (), probed identical to torch.cummin).


def _run(inp, dim, dtype=None):
    values = torch.empty(inp.shape, dtype=dtype or inp.dtype)
    indices = torch.empty(inp.shape, dtype=torch.int64)
    _cummin_helper(inp, values, indices, dim)
    return values, indices


@pytest.mark.parametrize("shape", [(4, 16), (33, 65), (2637,)])
def test_cummin_helper_last_dim_matches_torch(shape):
    torch.manual_seed(0)
    inp = torch.randn(shape)
    values, indices = _run(inp, -1)
    ref_v, ref_i = torch.cummin(inp, -1)
    torch.testing.assert_close(values, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


def test_cummin_helper_matches_aten_helper():
    torch.manual_seed(1)
    inp = torch.randn(8, 24)
    values = torch.empty(8, 24)
    indices = torch.empty(8, 24, dtype=torch.int64)
    _cummin_helper(inp, values, indices, 1)
    ref_v = torch.empty(8, 24)
    ref_i = torch.empty(8, 24, dtype=torch.int64)
    torch.ops.aten._cummin_helper(inp, ref_v, ref_i, 1)
    torch.testing.assert_close(values, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
def test_cummin_helper_dim_variants(dim):
    torch.manual_seed(2)
    inp = torch.randn(6, 9)
    values, indices = _run(inp, dim)
    ref_v, ref_i = torch.cummin(inp, dim)
    torch.testing.assert_close(values, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


def test_cummin_helper_3d_middle_dim():
    torch.manual_seed(3)
    inp = torch.randn(4, 33, 5)
    values, indices = _run(inp, 1)
    ref_v, ref_i = torch.cummin(inp, 1)
    torch.testing.assert_close(values, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.int64, torch.int32]
)
def test_cummin_helper_dtypes(dtype):
    torch.manual_seed(4)
    if dtype in (torch.int64, torch.int32):
        inp = torch.randint(-3, 3, (5, 17)).to(dtype)
    else:
        inp = torch.randn(5, 17).to(dtype)
    values, indices = _run(inp, -1)
    ref_v, ref_i = torch.cummin(inp, -1)
    torch.testing.assert_close(values, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


def test_cummin_helper_nan_propagates():
    # cummin semantics: once NaN is seen it is the running minimum; NaN
    # positions and the values/indices after them match torch.cummin.
    torch.manual_seed(5)
    inp = torch.randn(4, 32)
    inp.flatten()[::7] = float("nan")
    values, indices = _run(inp, -1)
    ref_v, ref_i = torch.cummin(inp, -1)
    torch.testing.assert_close(values, ref_v, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


def test_cummin_helper_ties_match_aten():
    # torch.cummin breaks ties toward the LATER index (probed on 2.15.0+cpu:
    # [[3., 1., 1., 2., 1.]] -> indices [[0, 1, 2, 2, 4]]); the kernel's
    # tie-break-right scan matches aten exactly.
    inp = torch.tensor([[3.0, 1.0, 1.0, 2.0, 1.0]])
    values, indices = _run(inp, -1)
    ref_v, ref_i = torch.cummin(inp, -1)
    assert indices[0].tolist() == ref_i[0].tolist()
    assert indices[0, 2].item() == 2 and indices[0, 4].item() == 4


def test_cummin_helper_length_one_dim():
    torch.manual_seed(6)
    inp = torch.randn(5, 1)
    values, indices = _run(inp, -1)
    torch.testing.assert_close(values, inp, rtol=0, atol=0)
    assert torch.all(indices == 0)


def test_cummin_helper_non_contiguous_input():
    torch.manual_seed(7)
    base = torch.randn(8, 32)
    inp = base.t()  # (32, 8), non-contiguous
    assert not inp.is_contiguous()
    values, indices = _run(inp.contiguous(), 0)
    ref_v, ref_i = torch.cummin(inp, 0)
    torch.testing.assert_close(values, ref_v, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


def test_cummin_helper_wider_output_dtype():
    # values.dtype != input.dtype forces the general path: fp16 input written
    # into an fp32 values buffer (compute dtype is fp32 per the wrapper).
    torch.manual_seed(8)
    inp = torch.randn(6, 33, dtype=torch.float16)
    values, indices = _run(inp, -1, dtype=torch.float32)
    ref_v, ref_i = torch.cummin(inp, -1)
    torch.testing.assert_close(values, ref_v.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)


def test_cummin_helper_invalid_dim_raises():
    inp = torch.randn(4, 8)
    values = torch.empty_like(inp)
    indices = torch.empty(inp.shape, dtype=torch.int64)
    with pytest.raises(AssertionError, match="Invalid dim"):
        _cummin_helper(inp, values, indices, 2)
    with pytest.raises(AssertionError, match="Invalid dim"):
        _cummin_helper(inp, values, indices, -3)


def test_cummin_helper_triton_copy_kernel_path(monkeypatch):
    # The general path (dtype mismatch) writes results back through the
    # module's own Triton copy kernel.
    mod = importlib.import_module("flaggems._cummin_helper")
    launches = []
    kernel = mod._copy_to_out_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    inp = torch.randn(6, 33, dtype=torch.float16)
    values, indices = _run(inp, -1, dtype=torch.float32)
    ref_v, ref_i = torch.cummin(inp, -1)
    assert len(launches) >= 1
    torch.testing.assert_close(values, ref_v.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(indices, ref_i, rtol=0, atol=0)
