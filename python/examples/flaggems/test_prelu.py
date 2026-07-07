import pytest
import torch
import torch.nn.functional as F

from .prelu import prelu


@pytest.mark.parametrize(
    "shape, num_channels",
    [
        ((512,), 1),
        ((512,), 512),
        ((2, 512), 1),
        ((2, 512), 512),
        ((2, 3, 32, 32), 1),
        ((2, 3, 32, 32), 3),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_prelu(shape, num_channels, dtype):
    if shape == (512,) and num_channels == 512:
        with pytest.raises(
            AssertionError,
            match="Per-channel PReLU weights require an input with an explicit channel dimension.",
        ):
            prelu(
                torch.randn(shape, dtype=dtype),
                torch.randn((num_channels,), dtype=dtype),
            )
        return

    a = torch.randn(shape, dtype=dtype)
    w = torch.randn((num_channels,), dtype=dtype)

    out_triton = prelu(a, w)
    out_torch = F.prelu(a, w)

    torch.testing.assert_close(out_triton, out_torch)
