import importlib

import pytest
import torch

from .fake_quantize_per_channel_affine import fake_quantize_per_channel_affine


def test_fq_per_channel_matches_aten_axis0():
    torch.manual_seed(0)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0, 1, 2], dtype=torch.int32)
    out = fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
    ref = torch.fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert out.dtype == x.dtype


def test_fq_per_channel_matches_aten_axis_variations():
    torch.manual_seed(1)
    x3d = torch.randn(2, 3, 4)
    for axis, n in ((0, 2), (1, 3), (2, 4)):
        scale = torch.full((n,), 0.1)
        zp = torch.arange(n, dtype=torch.int32)
        out = fake_quantize_per_channel_affine(x3d, scale, zp, axis, -128, 127)
        ref = torch.fake_quantize_per_channel_affine(x3d, scale, zp, axis, -128, 127)
        torch.testing.assert_close(out, ref, rtol=0, atol=0, msg=f"axis={axis}")
    x1d = torch.randn(6)
    scale = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15, 0.4])
    zp = torch.tensor([0, -1, 2, 3, 1, 0], dtype=torch.int32)
    out = fake_quantize_per_channel_affine(x1d, scale, zp, 0, -128, 127)
    torch.testing.assert_close(
        out,
        torch.fake_quantize_per_channel_affine(x1d, scale, zp, 0, -128, 127),
        rtol=0,
        atol=0,
    )


def test_fq_per_channel_uint8_range_and_per_channel_scales():
    torch.manual_seed(2)
    x = torch.randn(4, 8)
    scale = torch.tensor([0.05, 0.1, 0.2, 0.3])
    zp = torch.tensor([128, 0, 255, 1], dtype=torch.int32)
    out = fake_quantize_per_channel_affine(x, scale, zp, 0, 0, 255)
    ref = torch.fake_quantize_per_channel_affine(x, scale, zp, 0, 0, 255)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_channel_saturation():
    x = torch.tensor([[100.0, -100.0, 0.05, -0.05, 1.0, 2.0]])
    scale = torch.tensor([0.1])
    zp = torch.tensor([0], dtype=torch.int32)
    out = fake_quantize_per_channel_affine(x, scale, zp, 0, 0, 255)
    ref = torch.fake_quantize_per_channel_affine(x, scale, zp, 0, 0, 255)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(out[0, :2], torch.tensor([25.5, 0.0]), rtol=0, atol=0)


def test_fq_per_channel_zero_point_edges_and_degenerate_range():
    torch.manual_seed(3)
    x = torch.randn(3, 5)
    scale = torch.tensor([0.1, 0.2, 0.3])
    for qmin, qmax, zp in (
        (-128, 127, torch.tensor([-128, 0, 127], dtype=torch.int32)),
        (0, 255, torch.tensor([0, 128, 255], dtype=torch.int32)),
        (5, 5, torch.tensor([5, 5, 5], dtype=torch.int32)),
    ):
        out = fake_quantize_per_channel_affine(x, scale, zp, 0, qmin, qmax)
        ref = torch.fake_quantize_per_channel_affine(x, scale, zp, 0, qmin, qmax)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_channel_extreme_scales():
    torch.manual_seed(4)
    x = torch.randn(3, 9)
    for scale in (torch.full((3,), 1e-4), torch.full((3,), 32.0)):
        zp = torch.zeros(3, dtype=torch.int32)
        out = fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
        ref = torch.fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_channel_float16():
    torch.manual_seed(5)
    x = torch.randn(3, 17, dtype=torch.float16)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0, 1, 2], dtype=torch.int32)
    out = fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
    ref = torch.fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_channel_non_contiguous():
    torch.manual_seed(6)
    xt = torch.randn(4, 6).t()
    assert not xt.is_contiguous()
    scale = torch.full((6,), 0.1)
    zp = torch.zeros(6, dtype=torch.int32)
    out = fake_quantize_per_channel_affine(xt, scale, zp, 0, -128, 127)
    ref = torch.fake_quantize_per_channel_affine(xt, scale, zp, 0, -128, 127)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_fq_per_channel_empty_tensor():
    x = torch.empty(0, 4)
    scale = torch.tensor([0.1, 0.2, 0.3, 0.4])
    zp = torch.zeros(4, dtype=torch.int32)
    out = fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
    assert out.shape == (0, 4)


def test_fq_per_channel_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="input must be a torch.Tensor"):
        fake_quantize_per_channel_affine(
            [1.0, 2.0],
            torch.tensor([0.1]),
            torch.tensor([0], dtype=torch.int32),
            0,
            -128,
            127,
        )


def test_fq_per_channel_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.fake_quantize_per_channel_affine")
    launches = []
    kernel = mod.fake_quantize_per_channel_affine_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    x = torch.randn(3, 17)
    scale = torch.tensor([0.1, 0.2, 0.3])
    zp = torch.tensor([0, 1, 2], dtype=torch.int32)
    out = fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127)
    assert len(launches) >= 1
    torch.testing.assert_close(
        out,
        torch.fake_quantize_per_channel_affine(x, scale, zp, 0, -128, 127),
        rtol=0,
        atol=0,
    )
