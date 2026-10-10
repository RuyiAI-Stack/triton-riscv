import torch

from .as_strided_copy import as_strided_copy, as_strided_copy_out


def test_as_strided_copy():
    x = torch.arange(16, dtype=torch.float32, device="cpu")

    out = as_strided_copy(x, (3, 3), (4, 1), storage_offset=1)
    ref = torch.as_strided(x, (3, 3), (4, 1), storage_offset=1).clone()

    torch.testing.assert_close(out, ref)


def test_as_strided_copy_out():
    x = torch.arange(16, dtype=torch.float32, device="cpu")
    out = torch.empty((3, 3), dtype=torch.float32, device="cpu")

    ret = as_strided_copy_out(x, (3, 3), (4, 1), storage_offset=1, out=out)

    assert ret is out
    torch.testing.assert_close(
        out, torch.as_strided(x, (3, 3), (4, 1), storage_offset=1)
    )
