import torch

from .view_as_complex_copy import view_as_complex_copy
from .view_as_real_copy import view_as_real_copy


def test_complex_roundtrip_both_directions():
    torch.manual_seed(0)
    for cdt in (torch.complex64, torch.complex128):
        z = torch.randn(2, 3, dtype=cdt)
        r = view_as_real_copy(z)
        assert r.shape == (2, 3, 2)
        back = view_as_complex_copy(r)
        torch.testing.assert_close(back, z, rtol=0, atol=0)


def test_complex_pair_layout_interleaved():
    # The real copy interleaves (real, imag) per element, matching aten.
    torch.manual_seed(0)
    z = torch.randn(3, dtype=torch.complex128)
    r = view_as_real_copy(z)
    flat = r.reshape(-1)
    torch.testing.assert_close(flat[0::2], z.real, rtol=0, atol=0)
    torch.testing.assert_close(flat[1::2], z.imag, rtol=0, atol=0)
