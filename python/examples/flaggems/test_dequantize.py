import torch

from .dequantize import dequantize


def test_dequantize():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    q = torch.quantize_per_tensor(x, scale=0.5, zero_point=2, dtype=torch.quint8)

    out = dequantize(q)
    ref = q.dequantize()

    torch.testing.assert_close(out, ref)


def test_dequantize_per_channel():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    q = torch.quantize_per_channel(
        x,
        scales=torch.tensor([0.5, 1.0], dtype=torch.float64),
        zero_points=torch.tensor([1, 2], dtype=torch.int64),
        axis=0,
        dtype=torch.qint8,
    )

    out = dequantize(q)
    ref = q.dequantize()

    torch.testing.assert_close(out, ref)


def test_dequantize_per_channel_axis_one():
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    q = torch.quantize_per_channel(
        x,
        scales=torch.tensor([0.25, 0.5, 1.5], dtype=torch.float64),
        zero_points=torch.tensor([0, 1, -2], dtype=torch.int64),
        axis=1,
        dtype=torch.qint8,
    )

    out = dequantize(q)
    ref = q.dequantize()

    torch.testing.assert_close(out, ref)
