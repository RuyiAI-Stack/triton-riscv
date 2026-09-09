import pytest
import torch

from .segment_reduce import (
    _segment_reduce_backward,
    _segment_reduce_backward_out,
    segment_reduce,
    segment_reduce_out,
)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_segment_reduce_sum_lengths(shape, dtype):
    torch.manual_seed(0)
    data = torch.randn(shape, dtype=dtype, device="cpu")
    size = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else shape
    lengths = []
    curr = 0
    while curr < size:
        step = torch.randint(1, 5, (1,)).item()
        if curr + step > size:
            step = size - curr
        lengths.append(step)
        curr += step
    lengths = torch.tensor(lengths, dtype=torch.int64, device="cpu")

    tri_out = segment_reduce(data, "sum", lengths=lengths)
    ref_out = torch.segment_reduce(data, "sum", lengths=lengths)

    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_segment_reduce_out(shape, dtype):
    torch.manual_seed(0)
    data = torch.randn(shape, dtype=dtype, device="cpu")
    size = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else shape
    lengths = []
    curr = 0
    while curr < size:
        step = torch.randint(1, 5, (1,)).item()
        if curr + step > size:
            step = size - curr
        lengths.append(step)
        curr += step
    lengths = torch.tensor(lengths, dtype=torch.int64, device="cpu")

    out_shape = list(shape)
    out_shape[0] = len(lengths)
    tri_out = torch.empty(out_shape, dtype=dtype, device="cpu")

    ret = segment_reduce_out(data, "mean", lengths=lengths, out=tri_out)

    assert ret is tri_out
    ref_out = torch.segment_reduce(data, "mean", lengths=lengths)
    torch.testing.assert_close(tri_out, ref_out)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_segment_reduce_backward(shape, dtype):
    torch.manual_seed(0)
    data = torch.randn(shape, dtype=dtype, device="cpu")
    size = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else shape
    lengths = []
    curr = 0
    while curr < size:
        step = torch.randint(1, 5, (1,)).item()
        if curr + step > size:
            step = size - curr
        lengths.append(step)
        curr += step
    lengths = torch.tensor(lengths, dtype=torch.int64, device="cpu")

    out_shape = list(shape)
    out_shape[0] = len(lengths)
    grad = torch.randn(out_shape, dtype=dtype, device="cpu")

    output = segment_reduce(data, "sum", lengths=lengths)
    tri_out = _segment_reduce_backward(grad, output, data, "sum", lengths=lengths)

    data_ref = data.clone().requires_grad_(True)
    output_ref = torch.segment_reduce(data_ref, "sum", lengths=lengths)
    output_ref.backward(grad)
    ref_out = data_ref.grad

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128), (512,), (1023, 64)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_segment_reduce_backward_out(shape, dtype):
    torch.manual_seed(0)
    data = torch.randn(shape, dtype=dtype, device="cpu")
    size = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else shape
    lengths = []
    curr = 0
    while curr < size:
        step = torch.randint(1, 5, (1,)).item()
        if curr + step > size:
            step = size - curr
        lengths.append(step)
        curr += step
    lengths = torch.tensor(lengths, dtype=torch.int64, device="cpu")

    out_shape = list(shape)
    out_shape[0] = len(lengths)
    grad = torch.randn(out_shape, dtype=dtype, device="cpu")

    output = segment_reduce(data, "mean", lengths=lengths)
    tri_out = torch.empty_like(data)

    ret = _segment_reduce_backward_out(
        grad, output, data, "mean", lengths=lengths, out=tri_out
    )

    data_ref = data.clone().requires_grad_(True)
    output_ref = torch.segment_reduce(data_ref, "mean", lengths=lengths)
    output_ref.backward(grad)
    ref_out = data_ref.grad

    assert ret is tri_out
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-5, atol=1e-5)
