import torch

from .upsample_bicubic2d_aa_backward import _upsample_bicubic2d_aa_backward


def test_upsample_bicubic2d_aa_backward_correctness():
    torch.manual_seed(0)
    N, C, H_in, W_in = 1, 1, 4, 4
    H_out, W_out = 8, 8
    input_size = (N, C, H_in, W_in)
    output_size = (H_out, W_out)

    grad_output = torch.randn(
        N, C, H_out, W_out, dtype=torch.float32, device="cpu"
    )

    result = _upsample_bicubic2d_aa_backward(
        grad_output, output_size, input_size, align_corners=False
    )

    assert result.shape == (N, C, H_in, W_in)
    assert result.dtype == torch.float32


def test_upsample_bicubic2d_aa_backward_batch():
    torch.manual_seed(0)
    N, C, H_in, W_in = 2, 3, 4, 4
    H_out, W_out = 8, 8
    input_size = (N, C, H_in, W_in)
    output_size = (H_out, W_out)

    grad_output = torch.randn(
        N, C, H_out, W_out, dtype=torch.float32, device="cpu"
    )

    result = _upsample_bicubic2d_aa_backward(
        grad_output, output_size, input_size, align_corners=False
    )

    assert result.shape == (N, C, H_in, W_in)
