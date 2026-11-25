import torch

from .log_normal_ import log_normal_


def test_log_normal_():
    out = torch.empty(8)
    log_normal_(out)
    assert torch.all(out > 0)
