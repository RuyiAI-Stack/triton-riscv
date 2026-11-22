import importlib

import torch

from .fake_quantize_per_channel_affine_cachemask_backward import (
    fake_quantize_per_channel_affine_cachemask_backward,
)


def _ref(grad, mask):
    return torch.ops.aten.fake_quantize_per_channel_affine_cachemask_backward(
        grad, mask
    )


def test_pc_cachemask_backward_matches_aten():
    torch.manual_seed(0)
    for shape in ((7,), (3, 17), (2, 3, 5)):
        grad = torch.randn(shape)
        mask = torch.rand(shape) > 0.5
        out = fake_quantize_per_channel_affine_cachemask_backward(grad, mask)
        ref = _ref(grad, mask)
        assert out.dtype == grad.dtype
        assert out.shape == grad.shape
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_pc_cachemask_backward_zero_where_mask_false():
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [False, True]])
    out = fake_quantize_per_channel_affine_cachemask_backward(grad, mask)
    torch.testing.assert_close(
        out, torch.tensor([[1.0, 0.0], [0.0, 4.0]]), rtol=0, atol=0
    )
    torch.testing.assert_close(out, _ref(grad, mask), rtol=0, atol=0)


def test_pc_cachemask_backward_all_false_and_all_true_masks():
    grad = torch.randn(4, 5)
    out = fake_quantize_per_channel_affine_cachemask_backward(
        grad, torch.zeros(4, 5, dtype=torch.bool)
    )
    torch.testing.assert_close(out, torch.zeros(4, 5), rtol=0, atol=0)
    out = fake_quantize_per_channel_affine_cachemask_backward(
        grad, torch.ones(4, 5, dtype=torch.bool)
    )
    torch.testing.assert_close(out, grad, rtol=0, atol=0)


def test_pc_cachemask_backward_float16():
    torch.manual_seed(1)
    grad = torch.randn(2, 17, dtype=torch.float16)
    mask = torch.rand(2, 17) > 0.5
    out = fake_quantize_per_channel_affine_cachemask_backward(grad, mask)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, _ref(grad, mask), rtol=0, atol=0)


def test_pc_cachemask_backward_non_contiguous():
    torch.manual_seed(2)
    grad = torch.randn(4, 6).t()
    mask = (torch.rand(4, 6) > 0.5).t()
    out = fake_quantize_per_channel_affine_cachemask_backward(grad, mask)
    torch.testing.assert_close(out, _ref(grad, mask), rtol=0, atol=0)


def test_pc_cachemask_backward_empty():
    out = fake_quantize_per_channel_affine_cachemask_backward(
        torch.empty(0), torch.empty(0, dtype=torch.bool)
    )
    assert out.shape == (0,)


def test_pc_cachemask_backward_triton_path(monkeypatch):
    mod = importlib.import_module(
        "flaggems.fake_quantize_per_channel_affine_cachemask_backward"
    )
    launches = []
    kernel = mod.fake_quantize_per_channel_affine_cachemask_backward_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    grad = torch.randn(3, 17)
    mask = torch.rand(3, 17) > 0.5
    out = fake_quantize_per_channel_affine_cachemask_backward(grad, mask)
    assert len(launches) >= 1
    torch.testing.assert_close(out, _ref(grad, mask), rtol=0, atol=0)
