import torch

from ._resize_output import _resize_output, _resize_output_


def test_resize_output_copy_preserves_prefix():
    x = torch.arange(6, dtype=torch.float32, device="cpu")

    out = _resize_output(x, [8], x.device)

    assert out.shape == (8,)
    torch.testing.assert_close(out[:6], x)


def test_resize_output_inplace_shape():
    x = torch.arange(6, dtype=torch.float32, device="cpu")

    out = _resize_output_(x, [2, 3], x.device)

    assert out is x
    assert tuple(x.shape) == (2, 3)
