import pytest
import torch

from .arange import arange, arange_start


@pytest.mark.parametrize("end", [0, 10, 512, 1023, 1024, 2049])
def test_arange(end):
    torch.manual_seed(0)

    ref_out = torch.arange(end)
    tri_out = arange(end)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "start, end, step",
    [(0, 10, 1), (2, 20, 3), (10, 0, -1), (1024, 0, -3), (1, 10, 20)],
)
def test_arange_start_int(start, end, step):
    torch.manual_seed(0)

    ref_out = torch.arange(start, end, step=step)
    tri_out = arange_start(start, end, step=step)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "start, end, step", [(0.0, 5.5, 0.5), (10.0, -2.0, -1.5)]
)
def test_arange_start_float(start, end, step):
    torch.manual_seed(0)

    ref_out = torch.arange(start, end, step=step, dtype=torch.float32)
    tri_out = arange_start(start, end, step=step, dtype=torch.float32)

    torch.testing.assert_close(tri_out, ref_out, rtol=1e-4, atol=1e-4)
