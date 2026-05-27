import pytest
import torch

from .quantile import quantile


@pytest.mark.parametrize("shape", [(10,), (4, 16), (2, 4, 1025)])
@pytest.mark.parametrize("q", [0.5, [0.25, 0.5, 0.75]])
@pytest.mark.parametrize("dim", [None, -1])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_quantile(shape, q, dim, interpolation):
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=torch.float32)
    if isinstance(q, list):
        q_val = torch.tensor(q)
    else:
        q_val = q

    out_triton = quantile(a, q_val, dim=dim, interpolation=interpolation)
    out_torch = torch.quantile(a, q_val, dim=dim, interpolation=interpolation)

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)
