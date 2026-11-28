import importlib

import pytest
import torch

from ._new_zeros_with_same_feature_meta import _new_zeros_with_same_feature_meta


def test_basic_zero_fill_contiguous_common_case():
    # Default self_num_batch_dims=0: only other's feature shape is used.
    self_t = torch.empty(2, 3)
    other = torch.empty(4, 5)
    out = _new_zeros_with_same_feature_meta(self_t, other)
    assert out.shape == (4, 5)
    assert out.dtype == other.dtype
    assert bool((out == 0).all())


def test_batch_dims_slice_of_self():
    self_t = torch.empty(2, 3, 4)
    other = torch.empty(5, 6)
    out = _new_zeros_with_same_feature_meta(self_t, other, self_num_batch_dims=2)
    assert out.shape == (2, 3, 5, 6)
    assert bool((out == 0).all())


def test_zero_dim_self_batch():
    self_t = torch.empty(())
    other = torch.empty(3, 4)
    out = _new_zeros_with_same_feature_meta(self_t, other, self_num_batch_dims=1)
    assert out.shape == (0, 3, 4)  # aten: a 0-dim self contributes a size-0 batch dim
    assert bool((out == 0).all())


def test_complex_and_bool_zeroed():
    self_t = torch.empty(2)
    other_c = torch.empty(3, dtype=torch.complex128)
    out_c = _new_zeros_with_same_feature_meta(self_t, other_c)
    assert out_c.dtype == torch.complex128
    assert bool((out_c == 0).all())
    other_b = torch.empty(3, dtype=torch.bool)
    out_b = _new_zeros_with_same_feature_meta(self_t, other_b)
    assert out_b.dtype == torch.bool
    assert not bool(out_b.any())


def test_strided_other_reproduces_layout():
    # A strided `other` (transpose view) must not hit the contiguous fast path.
    other = torch.empty(4, 6, dtype=torch.float64).t()
    assert not other.is_contiguous()
    self_t = torch.empty(2)
    out = _new_zeros_with_same_feature_meta(self_t, other, self_num_batch_dims=1)
    assert out.shape == (2, 6, 4)
    assert bool((out == 0).all())


def test_type_errors():
    with pytest.raises(TypeError, match="must be torch.Tensor"):
        _new_zeros_with_same_feature_meta("nope", torch.empty(3))
    with pytest.raises(TypeError, match="self_num_batch_dims must be an int"):
        _new_zeros_with_same_feature_meta(
            torch.empty(2), torch.empty(3), self_num_batch_dims=True
        )
    with pytest.raises(RuntimeError, match="mismatch in length"):
        _new_zeros_with_same_feature_meta(
            torch.empty(2), torch.empty(3), self_num_batch_dims=-1
        )


def test_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._new_zeros_with_same_feature_meta")
    launches = []
    real_run = mod._zero_fill_contig.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._zero_fill_contig, "run", spy_run)
    out = _new_zeros_with_same_feature_meta(torch.empty(2), torch.empty(4, 5))
    assert len(launches) >= 1
    assert out.shape == (4, 5)
    assert bool((out == 0).all())
