import pytest
import torch

from .act_quant import act_quant_triton


def act_quant_ref(x, block_size, scale_fmt=None):
    N = x.size(-1)
    x_2d = x.view(-1, N).float()
    M = x_2d.size(0)
    n_blocks = N // block_size

    y_ref = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    s_ref = torch.empty((M, n_blocks), dtype=torch.float32, device=x.device)

    FP8_MAX = 448.0
    for i in range(M):
        for j in range(n_blocks):
            block = x_2d[i, j * block_size : (j + 1) * block_size]
            amax = torch.max(torch.abs(block))
            amax = torch.maximum(amax, torch.tensor(1e-4, device=x.device))

            if scale_fmt is not None:
                scale_raw = amax / FP8_MAX
                log2_scale = torch.log2(scale_raw)
                log2_ceil = torch.ceil(log2_scale)
                scale = torch.exp2(log2_ceil)
            else:
                scale = amax / FP8_MAX

            y_block = block / scale
            y_block = torch.clamp(y_block, -FP8_MAX, FP8_MAX)

            y_ref[i, j * block_size : (j + 1) * block_size] = y_block.to(
                torch.float8_e4m3fn
            )
            s_ref[i, j] = scale

    return y_ref.view(x.shape), s_ref.view(*x.shape[:-1], n_blocks)


@pytest.mark.parametrize("shape", [(16, 256), (4, 128)])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
def test_act_quant(shape, block_size, scale_fmt):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cpu")

    y_ref, s_ref = act_quant_ref(x, block_size=block_size, scale_fmt=scale_fmt)
    y_triton, s_triton = act_quant_triton(x, block_size=block_size, scale_fmt=scale_fmt)

    # float8_e4m3fn doesn't support allclose directly well, so convert to float32
    torch.testing.assert_close(y_ref.float(), y_triton.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(s_ref, s_triton, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("block_size", [64, 128])
def test_act_quant_midpoint_rounds_like_torch(block_size):
    x = torch.zeros((1, block_size), dtype=torch.bfloat16, device="cpu")
    x[0, 0] = torch.tensor(2.625, dtype=torch.bfloat16)
    x[0, 1] = torch.tensor(0.181640625, dtype=torch.bfloat16)

    y_ref, s_ref = act_quant_ref(x, block_size=block_size, scale_fmt=None)
    y_triton, s_triton = act_quant_triton(x, block_size=block_size, scale_fmt=None)

    assert s_ref[0, 0].item() == pytest.approx(0.005859375)
    assert s_triton[0, 0].item() == pytest.approx(0.005859375)
    assert y_ref.view(torch.uint8)[0, 1].item() == 0x60
    assert y_triton.view(torch.uint8)[0, 1].item() == 0x60
    torch.testing.assert_close(y_ref.float(), y_triton.float(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(s_ref, s_triton, rtol=0.0, atol=0.0)
