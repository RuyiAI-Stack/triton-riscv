import torch

from .linalg_matrix_exp import linalg_matrix_exp


def matrix_exp_backward(self, grad):
    n = self.shape[-1]
    block = torch.zeros(
        self.shape[:-2] + (2 * n, 2 * n), dtype=self.dtype, device=self.device
    )
    block[..., :n, :n] = self.transpose(-1, -2)
    block[..., n:, n:] = self.transpose(-1, -2)
    block[..., :n, n:] = grad.transpose(-1, -2)
    exp_block = linalg_matrix_exp(block)
    return exp_block[..., :n, n:].transpose(-1, -2).contiguous()
