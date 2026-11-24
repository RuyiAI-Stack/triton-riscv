import torch

from .unbind_copy import unbind_copy


def test_unbind_copy():
    x = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 3, 4)

    out = unbind_copy(x, dim=1)
    ref = torch.unbind(x, dim=1)

    assert len(out) == len(ref)
    for actual, expected in zip(out, ref):
        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() != x.data_ptr()
