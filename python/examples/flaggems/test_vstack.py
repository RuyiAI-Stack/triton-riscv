import math

import pytest
import torch

from .vstack import vstack


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vstack(size):
    torch.manual_seed(0)
    a = torch.randn(2, size, device="cpu", dtype=torch.float32)
    b = torch.randn(3, size, device="cpu", dtype=torch.float32)

    ref_out = torch.vstack([a, b])
    tri_out = vstack([a, b])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vstack_single(size):
    torch.manual_seed(0)
    a = torch.randn(4, size, device="cpu", dtype=torch.float32)

    ref_out = torch.vstack([a])
    tri_out = vstack([a])

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_vstack_many(size):
    torch.manual_seed(0)
    tensors = [
        torch.randn(1, size, device="cpu", dtype=torch.float32) for _ in range(6)
    ]

    ref_out = torch.vstack(tensors)
    tri_out = vstack(tensors)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_vstack_mixed_empty_tensors():
    torch.manual_seed(0)
    tensors = [
        torch.randn(0, 16, device="cpu", dtype=torch.float32),
        torch.randn(2, 16, device="cpu", dtype=torch.float32),
        torch.randn(0, 16, device="cpu", dtype=torch.float32),
        torch.randn(3, 16, device="cpu", dtype=torch.float32),
        torch.randn(1, 16, device="cpu", dtype=torch.float32),
    ]

    ref_out = torch.vstack(tensors)
    tri_out = vstack(tensors)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


def test_vstack_empty_output():
    tensors = [
        torch.randn(2, 0, device="cpu", dtype=torch.float32),
        torch.randn(3, 0, device="cpu", dtype=torch.float32),
    ]

    ref_out = torch.vstack(tensors)
    tri_out = vstack(tensors)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("trailing_shape", [(16,), (2, 8), (1025,)])
@pytest.mark.parametrize("stride_kind", ["zero", "small", "large"])
@pytest.mark.parametrize("rows", [(1,), (1, 2, 0, 1, 3, 1)])
def test_vstack_singleton_row_stride(trailing_shape, stride_kind, rows, monkeypatch):
    row_width = math.prod(trailing_shape)
    leading_stride = {"zero": 0, "small": 2, "large": 2 * row_width}[stride_kind]
    tensors = []
    for i, num_rows in enumerate(rows):
        # Pad storage so a regressed kernel can be checked without corrupting memory.
        backing = torch.arange(2 * max(num_rows, 1) * row_width, dtype=torch.float32)
        backing += i * 10000
        tensor = backing[: num_rows * row_width].reshape(num_rows, *trailing_shape)
        if num_rows == 1:
            tensor = tensor.as_strided(
                tensor.shape, (leading_stride, *tensor.stride()[1:])
            )
            assert tensor.is_contiguous()
            assert tensor.contiguous().stride(0) == leading_stride
        tensors.append(tensor)

    ref_out = torch.vstack(tensors)
    numel = ref_out.numel()
    sentinel = -999.0
    guarded_output = torch.full((2 * numel + 2,), sentinel)
    output = guarded_output[1 : numel + 1].view(ref_out.shape)
    original_empty = torch.empty

    def padded_empty(size, *args, **kwargs):
        if isinstance(size, (list, tuple)) and tuple(size) == tuple(ref_out.shape):
            return output
        return original_empty(size, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", padded_empty)
    tri_out = vstack(tensors)

    assert tri_out.data_ptr() == output.data_ptr()
    torch.testing.assert_close(guarded_output[:1], torch.full((1,), sentinel))
    torch.testing.assert_close(
        guarded_output[numel + 1 :], torch.full((numel + 1,), sentinel)
    )
    torch.testing.assert_close(tri_out, ref_out, rtol=0, atol=0)
