import pytest
import torch

from .groupnorm import group_norm


@pytest.mark.parametrize("size", [512, 1023, 1024])
def test_groupnorm_forward(size):
    """Test group_norm forward against torch.nn.functional.group_norm."""
    torch.manual_seed(0)
    N, C, H, W = 2, 6, size, 1
    HxW = H * W
    num_groups = 3
    input = torch.randn(N, C, HxW, device="cpu", dtype=torch.float32)
    weight = torch.randn(C, device="cpu", dtype=torch.float32)
    bias = torch.randn(C, device="cpu", dtype=torch.float32)

    ref = torch.nn.functional.group_norm(input, num_groups, weight, bias, eps=1e-5)
    y, mean, rstd = group_norm(input, weight, bias, N, C, HxW, num_groups)

    torch.testing.assert_close(y[0], ref[0], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1024])
def test_groupnorm_backward(size):
    torch.manual_seed(0)
    N, C, H, W = 2, 6, size, 1
    HxW = H * W
    num_groups = 3
    input = torch.randn(
        N, C, HxW, device="cpu", dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(C, device="cpu", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(C, device="cpu", dtype=torch.float32, requires_grad=True)

    # Reference backward via torch
    ref_out = torch.nn.functional.group_norm(input, num_groups, weight, bias, eps=1e-5)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    ref_dx = input.grad.clone()
    ref_dw = weight.grad.clone()
    ref_db = bias.grad.clone()

    # Triton backward
    input.grad = None
    weight.grad = None
    bias.grad = None
    from .groupnorm import group_norm_backward

    y, mean, rstd = group_norm(input, weight, bias, N, C, HxW, num_groups)
    tri_dx, tri_dw, tri_db = group_norm_backward(
        grad_out,
        input,
        mean,
        rstd,
        weight,
        N,
        C,
        HxW,
        num_groups,
        output_mask=(True, True, True),
    )

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_dw, ref_dw, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_db, ref_db, rtol=1e-3, atol=1e-3)


def test_groupnorm_no_weight_bias():
    """Test group_norm without weight and bias."""
    torch.manual_seed(0)
    N, C, H, W = 2, 6, 512, 1
    HxW = H * W
    num_groups = 3
    input = torch.randn(N, C, HxW, device="cpu", dtype=torch.float32)

    ref = torch.nn.functional.group_norm(input, num_groups, None, None, eps=1e-5)
    y, mean, rstd = group_norm(input, None, None, N, C, HxW, num_groups)

    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [512, 1024])
def test_groupnorm_different_groups(size):
    """Test group_norm with different numbers of groups."""
    torch.manual_seed(0)
    N, C, H, W = 1, 8, size, 1
    HxW = H * W
    input = torch.randn(N, C, HxW, device="cpu", dtype=torch.float32)

    for num_groups in [2, 4, 8]:
        ref = torch.nn.functional.group_norm(input, num_groups, eps=1e-5)
        y, mean, rstd = group_norm(input, None, None, N, C, HxW, num_groups)
        torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)
