import pytest
import torch
import torch.nn.functional as F

from .batch_norm import batch_norm


@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (4, 8, 16)])
@pytest.mark.parametrize("training", [False, True])
def test_batch_norm_forward(shape, training):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    feat_dim = shape[1]

    weight = torch.randn(feat_dim, dtype=torch.float32, device="cpu")
    bias = torch.randn(feat_dim, dtype=torch.float32, device="cpu")
    running_mean = torch.zeros(feat_dim, dtype=torch.float32, device="cpu")
    running_var = torch.ones(feat_dim, dtype=torch.float32, device="cpu")

    ref_out = F.batch_norm(
        x, running_mean, running_var, weight, bias, training=training
    )
    rm_ref = running_mean.clone()
    rv_ref = running_var.clone()

    running_mean.zero_()
    running_var.fill_(1)

    tri_out, tri_mean, tri_inv_std = batch_norm(
        x,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)

    if training:
        torch.testing.assert_close(running_mean, rm_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(running_var, rv_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 4, 8, 8)])
@pytest.mark.parametrize("training", [False, True])
def test_batch_norm_backward(shape, training):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu", requires_grad=True)
    feat_dim = shape[1]

    weight = torch.randn(
        feat_dim, dtype=torch.float32, device="cpu", requires_grad=True
    )
    bias = torch.randn(feat_dim, dtype=torch.float32, device="cpu", requires_grad=True)
    running_mean = torch.zeros(feat_dim, dtype=torch.float32, device="cpu")
    running_var = torch.ones(feat_dim, dtype=torch.float32, device="cpu")

    # Reference backward via torch
    ref_out = F.batch_norm(
        x, running_mean, running_var, weight, bias, training=training
    )
    grad_out = torch.randn_like(ref_out)
    ref_out.backward(grad_out)
    ref_dx = x.grad.clone()
    ref_dw = weight.grad.clone()
    ref_db = bias.grad.clone()

    # Triton backward
    x.grad = None
    weight.grad = None
    bias.grad = None
    running_mean.zero_()
    running_var.fill_(1)

    tri_out, save_mean, save_invstd = batch_norm(
        x,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
    )
    from .batch_norm import batch_norm_backward

    tri_dx, tri_dw, tri_db = batch_norm_backward(
        grad_out,
        x,
        weight=weight,
        running_mean=running_mean,
        running_var=running_var,
        save_mean=save_mean,
        save_invstd=save_invstd,
        train=training,
        eps=1e-5,
        output_mask=(True, True, True),
    )

    torch.testing.assert_close(tri_dx, ref_dx, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_dw, ref_dw, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(tri_db, ref_db, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(2, 4, 8, 8)])
def test_batch_norm_2d_input(shape):
    torch.manual_seed(0)
    x_4d = torch.randn(shape, dtype=torch.float32, device="cpu")
    x_2d = x_4d[:, :, 0, 0]
    feat_dim = shape[1]

    weight = torch.randn(feat_dim, dtype=torch.float32, device="cpu")
    bias = torch.randn(feat_dim, dtype=torch.float32, device="cpu")
    running_mean = torch.zeros(feat_dim, dtype=torch.float32, device="cpu")
    running_var = torch.ones(feat_dim, dtype=torch.float32, device="cpu")

    ref_out = F.batch_norm(
        x_2d, running_mean, running_var, weight, bias, training=False
    )
    tri_out, _, _ = batch_norm(
        x_2d,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=False,
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 4, 8, 8)])
def test_batch_norm_no_affine(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device="cpu")
    feat_dim = shape[1]
    running_mean = torch.zeros(feat_dim, dtype=torch.float32, device="cpu")
    running_var = torch.ones(feat_dim, dtype=torch.float32, device="cpu")

    ref_out = F.batch_norm(x, running_mean, running_var, None, None, training=False)
    tri_out, _, _ = batch_norm(
        x, running_mean=running_mean, running_var=running_var, training=False
    )

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
