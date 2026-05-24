import pytest
import torch

from .one_hot import one_hot


@pytest.mark.parametrize("shape", [(512,), (1023,), (1024,)])
def test_one_hot(shape):
    torch.manual_seed(0)
    x = torch.randint(0, 10, shape, device="cpu")
    num_classes = int(x.max().item()) + 1

    ref = torch.nn.functional.one_hot(x, num_classes)
    tri = one_hot(x, num_classes)

    torch.testing.assert_close(tri, ref, rtol=0, atol=0)
