import importlib

import pytest
import torch
import torch.nn.functional as F

from ._pad_enum import _pad_enum

# torch._C._nn._pad_enum numbering (verified against aten): 0 = reflect,
# 1 = replicate, 2 = circular, 3 = constant.
REFLECT, REPLICATE, CIRCULAR, CONSTANT = 0, 1, 2, 3


@pytest.mark.parametrize(
    "shape, pad, mode, mode_name",
    [
        ((3, 7), (1, 2), REFLECT, "reflect"),
        ((2, 3, 6), (2, 1, 1, 0), REFLECT, "reflect"),
        ((2, 2, 3, 5), (1, 1, 1, 1, 1, 1), REFLECT, "reflect"),
        ((3, 7), (0, 3), REPLICATE, "replicate"),
        ((2, 3, 6), (2, 0, 1, 2), REPLICATE, "replicate"),
        ((2, 2, 3, 5), (1, 2, 1, 0, 0, 1), REPLICATE, "replicate"),
        ((3, 7), (2, 2), CIRCULAR, "circular"),
        ((2, 3, 6), (1, 2, 0, 1), CIRCULAR, "circular"),
        ((2, 2, 3, 5), (1, 1, 1, 1, 1, 1), CIRCULAR, "circular"),
    ],
)
def test_pad_enum_nonconstant_matches_fpad(shape, pad, mode, mode_name):
    x = torch.randn(shape, dtype=torch.float64)
    out = _pad_enum(x, pad, mode)
    ref = F.pad(x, pad, mode=mode_name)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    # and against aten's own _pad_enum for the dispatched combinations
    ref2 = torch._C._nn._pad_enum(x, pad, mode)
    torch.testing.assert_close(out, ref2, rtol=0, atol=0)


@pytest.mark.parametrize("value", [None, 0.0, 9.5, -1.25])
def test_pad_enum_constant_matches_fpad(value):
    x = torch.randn(2, 3, 6, dtype=torch.float64)
    out = _pad_enum(x, (1, 2, 0, 1), CONSTANT, value)
    ref = F.pad(x, (1, 2, 0, 1), mode="constant", value=value if value else 0.0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_enum_constant_negative_padding_crops_like_aten():
    x = torch.arange(2 * 6, dtype=torch.float64).reshape(2, 6)
    out = _pad_enum(x, (-2, -1), CONSTANT, None)
    ref = F.pad(x, (-2, -1), mode="constant")
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_enum_reflect_size1_dim_with_zero_pad():
    # The kernel's in_shape[d] == 1 reflect branch: the last dim has size 1
    # and zero padding on it; only H is padded. aten needs a batch dim for
    # a 4-length pad, so the oracle runs on the unsqueezed tensor.
    x = torch.arange(3, dtype=torch.float64).reshape(3, 1)
    out = _pad_enum(x, (0, 0, 1, 1), REFLECT)
    ref = F.pad(x.unsqueeze(0), (0, 0, 1, 1), mode="reflect").squeeze(0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_enum_int_dtypes():
    xi = torch.arange(2 * 5, dtype=torch.int64).reshape(2, 5)
    torch.testing.assert_close(
        _pad_enum(xi, (1, 1), REPLICATE),
        F.pad(xi, (1, 1), mode="replicate"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _pad_enum(xi, (1, 1), CONSTANT, 7), F.pad(xi, (1, 1), value=7), rtol=0, atol=0
    )


def test_pad_enum_1d_input_all_modes():
    # aten rejects non-constant F.pad on 1-D input, so the oracle runs the
    # same pad on a (1, W) batched tensor (upstream supports 1-D directly).
    x = torch.arange(6, dtype=torch.float64)
    for mode, name in (
        (REFLECT, "reflect"),
        (REPLICATE, "replicate"),
        (CIRCULAR, "circular"),
    ):
        out = _pad_enum(x, (1, 2), mode)
        ref = F.pad(x.unsqueeze(0), (1, 2), mode=name).squeeze(0)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pad_enum_negative_constant_crop():
    x = torch.arange(2 * 8, dtype=torch.float64).reshape(2, 8)
    out = _pad_enum(x, (-2, -2), CONSTANT, None)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, x[:, 2:6], rtol=0, atol=0)
    out2 = _pad_enum(torch.empty(0, 3), (1, 1), CONSTANT, None)
    assert out2.shape == (0, 5)
    out3 = _pad_enum(torch.randn(2, 4), (-2, -2), CONSTANT, None)
    assert out3.shape == (2, 0)


def test_pad_enum_assert_errors():
    x = torch.randn(2, 4)
    with pytest.raises(AssertionError, match="divisible by 2"):
        _pad_enum(x, (1, 2, 3), CONSTANT)
    with pytest.raises(AssertionError, match="Padding length too large"):
        _pad_enum(x, (1, 1, 1, 1, 1, 1), CONSTANT)
    with pytest.raises(
        AssertionError, match="less than the corresponding input dimension"
    ):
        _pad_enum(x, (4, 0), REFLECT)
    with pytest.raises(AssertionError, match="wrapping around more than once"):
        _pad_enum(x, (5, 0), CIRCULAR)


def test_pad_enum_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._pad_enum")
    launches = []
    kernel = mod._pad_enum_kernel
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(2, 3, 7, dtype=torch.float64)
    for mode, name in (
        (REFLECT, "reflect"),
        (REPLICATE, "replicate"),
        (CIRCULAR, "circular"),
    ):
        out = _pad_enum(x, (1, 2, 0, 1), mode)
        torch.testing.assert_close(
            out, F.pad(x, (1, 2, 0, 1), mode=name), rtol=0, atol=0
        )
    assert len(launches) >= 3
