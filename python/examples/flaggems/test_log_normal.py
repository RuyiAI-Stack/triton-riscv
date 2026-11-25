import torch

from .log_normal import log_normal


def test_log_normal():
    out = log_normal(8)
    assert torch.all(out > 0)
