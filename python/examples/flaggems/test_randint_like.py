import torch

from .randint_like import randint_like


def test_randint_like():
    like = torch.empty(8, dtype=torch.int64)
    out = randint_like(like, 3)
    assert out.shape == like.shape and int(out.max()) < 3
