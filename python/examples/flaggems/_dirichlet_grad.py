from .detach_copy import _copy_values


def _dirichlet_grad(x, alpha, grad_output):
    _ = alpha
    total = (grad_output * x).sum(dim=-1, keepdim=True)
    return _copy_values(grad_output - total)
