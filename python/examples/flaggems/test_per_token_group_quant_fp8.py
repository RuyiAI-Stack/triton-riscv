import pytest
import torch

from .per_token_group_quant_fp8 import per_token_group_quant_fp8


def ref_per_token_group_quant_fp8(
    x,
    group_size,
    eps=1e-10,
    dtype=None,
    column_major_scales=False,
    scale_ue8m0=False,
):
    fp8_dtype = dtype if dtype is not None else torch.float32
    if fp8_dtype == torch.float32:
        fp8_min = -256.0
        fp8_max = 256.0
    else:
        finfo = torch.finfo(fp8_dtype)
        fp8_min = finfo.min
        fp8_max = finfo.max

    x_reshaped = x.reshape(-1, group_size)
    _absmax = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True)
    _absmax = torch.maximum(_absmax, torch.tensor(eps, dtype=_absmax.dtype))

    y_s = _absmax / fp8_max

    if scale_ue8m0:
        y_s = torch.exp2(
            torch.ceil(
                torch.log2(
                    torch.maximum(
                        torch.abs(y_s), torch.tensor(1e-10, dtype=y_s.dtype)
                    )
                )
            )
        )

    y_q = torch.clamp(x_reshaped / y_s, fp8_min, fp8_max).to(fp8_dtype)

    x_q = y_q.reshape_as(x)
    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s_out = torch.empty(
            shape, device=x.device, dtype=torch.float32
        ).permute(*range(1, x.ndim), 0)
        x_s_out.copy_(y_s.reshape(*x.shape[:-1], x.shape[-1] // group_size))
        x_s = x_s_out
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = y_s.reshape(shape)

    return x_q, x_s


@pytest.mark.parametrize(
    "shape, group_size",
    [
        ((32, 128), 32),
        ((16, 256), 64),
        ((2, 64, 128), 64),
    ],
)
@pytest.mark.parametrize("column_major_scales", [False, True])
@pytest.mark.parametrize("scale_ue8m0", [False, True])
def test_per_token_group_quant_fp8(
    shape, group_size, column_major_scales, scale_ue8m0
):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)

    tri_q, tri_s = per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major_scales,
        scale_ue8m0=scale_ue8m0,
        dtype=torch.float32,
    )
    ref_q, ref_s = ref_per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major_scales,
        scale_ue8m0=scale_ue8m0,
        dtype=torch.float32,
    )

    torch.testing.assert_close(tri_q, ref_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_s, ref_s, rtol=1e-3, atol=1e-3)
