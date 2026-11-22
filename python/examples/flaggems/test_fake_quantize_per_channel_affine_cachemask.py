import importlib

import torch

from .fake_quantize_per_channel_affine_cachemask import (
    fake_quantize_per_channel_affine_cachemask,
    fake_quantize_per_channel_affine_cachemask_out,
)


def _ref(x, scale, zp, axis, qmin, qmax):
    return torch.ops.aten.fake_quantize_per_channel_affine_cachemask(
        x, scale, zp, axis, qmin, qmax
    )


def test_pc_cachemask_returns_out_and_bool_mask():
    torch.manual_seed(0)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0, 1, 2], dtype=torch.int32)
    result = fake_quantize_per_channel_affine_cachemask(x, scale, zp, 0, -128, 127)
    assert isinstance(result, tuple) and len(result) == 2
    out, mask = result
    ref_out, ref_mask = _ref(x, scale, zp, 0, -128, 127)
    assert mask.dtype == torch.bool
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_pc_cachemask_axis_variations():
    torch.manual_seed(1)
    x3d = torch.randn(2, 3, 4)
    for axis, n in ((0, 2), (1, 3), (2, 4)):
        scale = torch.full((n,), 0.1)
        zp = torch.arange(n, dtype=torch.int32)
        out, mask = fake_quantize_per_channel_affine_cachemask(
            x3d, scale, zp, axis, -128, 127
        )
        ref_out, ref_mask = _ref(x3d, scale, zp, axis, -128, 127)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0, msg=f"axis={axis}")
        torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_pc_cachemask_mask_semantics_saturation_and_nan():
    x = torch.tensor([[100.0, -100.0], [0.05, -0.05]])
    scale = torch.tensor([0.1, 0.1])
    zp = torch.tensor([0, 0], dtype=torch.int32)
    out, mask = fake_quantize_per_channel_affine_cachemask(x, scale, zp, 1, 0, 255)
    ref_out, ref_mask = _ref(x, scale, zp, 1, 0, 255)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
    assert mask.tolist() == [[False, False], [True, True]]
    nan = torch.tensor([[float("nan"), 1.0]])
    out, mask = fake_quantize_per_channel_affine_cachemask(
        nan, torch.tensor([0.1]), torch.tensor([0], dtype=torch.int32), 0, 0, 255
    )
    ref_out, ref_mask = _ref(
        nan, torch.tensor([0.1]), torch.tensor([0], dtype=torch.int32), 0, 0, 255
    )
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0, equal_nan=True)
    assert mask.tolist() == [[False, True]]


def test_pc_cachemask_zero_point_edges_and_degenerate_range():
    torch.manual_seed(2)
    x = torch.randn(3, 5)
    scale = torch.tensor([0.1, 0.2, 0.3])
    for qmin, qmax, zp in (
        (-128, 127, torch.tensor([-128, 0, 127], dtype=torch.int32)),
        (0, 255, torch.tensor([0, 128, 255], dtype=torch.int32)),
        (5, 5, torch.tensor([5, 5, 5], dtype=torch.int32)),
    ):
        out, mask = fake_quantize_per_channel_affine_cachemask(
            x, scale, zp, 0, qmin, qmax
        )
        ref_out, ref_mask = _ref(x, scale, zp, 0, qmin, qmax)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_pc_cachemask_float16():
    torch.manual_seed(3)
    x = torch.randn(3, 17, dtype=torch.float16)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0, 1, 2], dtype=torch.int32)
    out, mask = fake_quantize_per_channel_affine_cachemask(x, scale, zp, 0, -128, 127)
    ref_out, ref_mask = _ref(x, scale, zp, 0, -128, 127)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_pc_cachemask_non_contiguous():
    torch.manual_seed(4)
    xt = torch.randn(4, 6).t()
    scale = torch.full((6,), 0.1)
    zp = torch.zeros(6, dtype=torch.int32)
    out, mask = fake_quantize_per_channel_affine_cachemask(xt, scale, zp, 0, -128, 127)
    ref_out, ref_mask = _ref(xt, scale, zp, 0, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)


def test_pc_cachemask_empty_tensor():
    x = torch.empty(0, 4)
    scale = torch.tensor([0.1, 0.2, 0.3, 0.4])
    zp = torch.zeros(4, dtype=torch.int32)
    out, mask = fake_quantize_per_channel_affine_cachemask(x, scale, zp, 0, -128, 127)
    assert out.shape == (0, 4) and mask.shape == (0, 4)
    assert mask.dtype == torch.bool


def test_pc_cachemask_out_writes_into_buffers():
    torch.manual_seed(5)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0, 1, 2], dtype=torch.int32)
    out0 = torch.empty(3, 17)
    out1 = torch.empty(3, 17, dtype=torch.bool)
    ret0, ret1 = fake_quantize_per_channel_affine_cachemask_out(
        x, scale, zp, 0, -128, 127, out0=out0, out1=out1
    )
    assert ret0 is out0 and ret1 is out1
    ref_out, ref_mask = _ref(x, scale, zp, 0, -128, 127)
    torch.testing.assert_close(out0, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(out1, ref_mask, rtol=0, atol=0)


def test_pc_cachemask_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fake_quantize_per_channel_affine_cachemask")
    launches = []
    kernel = mod.fake_quantize_per_channel_affine_cachemask_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0, 1, 2], dtype=torch.int32)
    out, mask = fake_quantize_per_channel_affine_cachemask(x, scale, zp, 0, -128, 127)
    assert len(launches) >= 1
    ref_out, ref_mask = _ref(x, scale, zp, 0, -128, 127)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
