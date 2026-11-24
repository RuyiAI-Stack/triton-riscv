import torch

from .matrix_exp_backward import matrix_exp_backward


def test_matrix_exp_backward():
    torch.manual_seed(0)
    raw = torch.randn(3, 3)
    a = raw @ raw.T + 0.5 * torch.eye(3)
    small = (a * 0.05).detach().requires_grad_()
    torch.linalg.matrix_exp(small).sum().backward()
    torch.testing.assert_close(
        matrix_exp_backward(small.detach(), torch.ones_like(small)),
        small.grad,
        rtol=1e-4,
        atol=1e-4,
    )
