import re
import subprocess

import pytest
import torch
import triton

from triton.backends.triton_shared.riscv import DEFAULT_LLC_FEATURES, DEFAULT_TRIPLE
from test_utils import get_llvm_bin_path, requires_rvv_execution

from ._euclidean_dist import _euclidean_dist, _euclidean_dist_kernel


@requires_rvv_execution
def ref_euclidean_dist(x1, x2):
    x1_norm = (x1**2).sum(dim=1, keepdim=True)
    x2_norm = (x2**2).sum(dim=1, keepdim=True)
    dist = x1_norm + x2_norm.T - 2.0 * torch.mm(x1, x2.T)
    return torch.sqrt(torch.clamp(dist, min=0.0))


@pytest.mark.parametrize("N, M, D", [(8, 10, 16), (16, 8, 32), (32, 64, 128)])
def test_euclidean_dist(N, M, D):
    torch.manual_seed(0)
    x1 = torch.randn(N, D, device="cpu", dtype=torch.float32)
    x2 = torch.randn(M, D, device="cpu", dtype=torch.float32)

    ref = torch.cdist(x1, x2)
    tri = _euclidean_dist(x1, x2)

    torch.testing.assert_close(tri, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("N, M, D", [(8, 10, 16), (16, 8, 32), (32, 64, 128)])
def test_euclidean_dist_kernel_emits_riscv_object(tmp_path, N, M, D):
    x1 = torch.empty((N, D), device="cpu", dtype=torch.float32)
    x2 = torch.empty((M, D), device="cpu", dtype=torch.float32)
    out = torch.empty((N, M), device="cpu", dtype=torch.float32)
    obj_path = tmp_path / "euclidean_dist.o"
    block_d = min(triton.next_power_of_2(D), 1024)

    compiled = _euclidean_dist_kernel.warmup(
        x1,
        x2,
        out,
        N,
        M,
        D,
        x1.stride(0),
        x2.stride(0),
        out.stride(0),
        BLOCK_D=block_d,
        grid=(N, M),
        target_triple=DEFAULT_TRIPLE,
        target_features=DEFAULT_LLC_FEATURES,
    )
    obj_path.write_bytes(compiled.asm["obj"])

    asm = subprocess.check_output(
        [get_llvm_bin_path("llvm-objdump"), "-d", str(obj_path)],
        text=True,
    )
    assert re.search(r"<_euclidean_dist_kernel>:", asm)
    assert re.search(r"\bvsetivli\b", asm)
    assert re.search(r"\bvle32\.v\b", asm)
    assert re.search(r"\bvse32\.v\b", asm)
    assert re.search(r"\bvfsub\.vv\b", asm)
    assert re.search(r"\bvfmul\.vv\b", asm)


def test_euclidean_dist_shape_assertions():
    x1 = torch.randn(8, 16)
    x2 = torch.randn(10, 32)

    with pytest.raises(AssertionError):
        _euclidean_dist(x1, x2)
