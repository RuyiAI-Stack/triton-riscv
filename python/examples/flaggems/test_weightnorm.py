import pytest
import torch

from .weightnorm import weight_norm_interface, weight_norm_interface_backward


def test_weight_norm_interface_last_dim():
    torch.manual_seed(0)
    v = torch.randn(4, 8, dtype=torch.float32, device="cpu")
    g = torch.randn(8, dtype=torch.float32, device="cpu")

    output, norm = weight_norm_interface(v, g, dim=1)

    assert output.shape == v.shape
    assert norm.shape == g.shape


def test_weight_norm_interface_first_dim():
    torch.manual_seed(0)
    v = torch.randn(4, 8, dtype=torch.float32, device="cpu")
    g = torch.randn(4, dtype=torch.float32, device="cpu")

    output, norm = weight_norm_interface(v, g, dim=0)

    assert output.shape == v.shape
    assert norm.shape == g.shape


def test_weight_norm_interface_1d():
    torch.manual_seed(0)
    v = torch.randn(16, dtype=torch.float32, device="cpu")
    g = torch.randn(16, dtype=torch.float32, device="cpu")

    output, norm = weight_norm_interface(v, g, dim=0)

    assert output.shape == v.shape
    assert norm.shape == g.shape


@pytest.mark.parametrize(
    "v_shape, g_shape, dim",
    [
        ((4, 8), (8,), 1),
        ((4, 8), (4,), 0),
    ],
)
def test_weight_norm_interface_backward(v_shape, g_shape, dim):
    torch.manual_seed(0)
    v = torch.randn(
        v_shape, dtype=torch.float32, device="cpu", requires_grad=True
    )
    g = torch.randn(
        g_shape, dtype=torch.float32, device="cpu", requires_grad=True
    )

    # Reference backward via manual formula to avoid private API changes in PyTorch
    v_norm = torch.norm_except_dim(v, 2, dim)
    g_expand = g.view(-1, 1) if dim == 0 else g
    ref_out = v * (g_expand / v_norm)
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    ref_dv = v.grad.clone()
    ref_dg = g.grad.clone()

    # Triton backward
    v.grad = None
    g.grad = None
    saved_v = v.detach().clone()
    saved_g = g.detach().clone()
    w_normalized, saved_norm = weight_norm_interface(saved_v, saved_g, dim=dim)
    tri_dv, tri_dg = weight_norm_interface_backward(
        grad_out, saved_v, saved_g, saved_norm, dim
    )

    torch.testing.assert_close(tri_dv, ref_dv, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_dg, ref_dg, rtol=1e-3, atol=1e-3)
