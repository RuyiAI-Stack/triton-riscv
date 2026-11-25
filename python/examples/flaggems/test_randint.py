import torch

from .randint import randint


def test_randint():
    out = randint(0, 5, (32,))
    assert out.dtype == torch.int64 and int(out.min()) >= 0 and int(out.max()) < 5
