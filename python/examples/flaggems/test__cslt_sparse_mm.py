import importlib

import pytest
import torch

from ._cslt_sparse_mm import _cslt_sparse_mm

# fp16 operands round at ~1e-3 relative on O(1) tiles; atol 1e-2 leaves
# margin for fp32 accumulation-order differences across the decompressed
# 2:4 tiles.

# oracle: formula per upstream _cslt_sparse_mm.py (aten op and
# torch._cslt_compress are CUDA-only, so there is no CPU oracle). The test
# HAND-ENCODES the cuSPARSELt compressed blob by inverting the upstream
# _decompress_kernel (lines 60-141): values are the 2:4 nonzeros row-major
# (M x K/2), and each group-of-4's kept column pair is stored as the 4-bit
# nibble idx0 | (idx1 << 2) at the tile-swizzled index
#   rt = row // 32, ri = row % 32, gt = goff // 8, gi = goff % 8,
#   col = (gi % 4) * 4 + (2 if gi >= 4 else 0),
#   within = (ri % 8) * 16 + ((ri // 8) % 2) + ((ri // 8) // 2) * 128 + col,
#   nib_idx = (rt % 2) * 256 + gt * 512 + (rt // 2) * (n_kt * 512) + within,
# packed 4 nibbles per uint16 word. FLAG: probe-dependent — the layout is
# the upstream Hopper decoder's own description, not an independent
# cuSPARSELt specification.


def _make_2to4(M, K, dtype):
    # Exactly two kept lanes per group of 4; vary the kept-position pattern to
    # stress every nibble value.
    a = torch.randn(M, K, dtype=torch.float32).clamp(-2, 2)
    dense = torch.zeros(M, K, dtype=dtype)
    patterns = [(0, 1), (2, 3), (0, 3), (1, 2)]
    for row in range(M):
        for goff in range(K // 4):
            i0, i1 = patterns[(row + goff) % 4]
            dense[row, goff * 4 + i0] = a[row, goff * 4 + i0].to(dtype)
            dense[row, goff * 4 + i1] = a[row, goff * 4 + i1].to(dtype)
    return dense.contiguous()


def _compress_2to4(dense):
    # Exact inverse of the upstream _decompress_kernel swizzle.
    M, K = dense.shape
    gpr = K // 4
    n_kt = (gpr + 7) // 8
    values = torch.zeros(M, K // 2, dtype=dense.dtype)
    layout_nibbles = ((M + 63) // 64) * n_kt * 512
    words = torch.zeros(layout_nibbles // 4, dtype=torch.int32)
    for row in range(M):
        rt, ri = row // 32, row % 32
        for goff in range(gpr):
            group = dense[row, goff * 4 : goff * 4 + 4]
            i0, i1 = (group != 0).nonzero().flatten().tolist()
            values[row, goff * 2] = group[i0]
            values[row, goff * 2 + 1] = group[i1]
            gt, gi = goff // 8, goff % 8
            col = (gi % 4) * 4 + (2 if gi >= 4 else 0)
            within = (ri % 8) * 16 + ((ri // 8) % 2) + ((ri // 8) // 2) * 128 + col
            nib_idx = (rt % 2) * 256 + gt * 512 + (rt // 2) * (n_kt * 512) + within
            nib = (i0 & 3) | ((i1 & 3) << 2)
            w, r = divmod(nib_idx, 4)
            words[w] |= nib << (4 * r)
    meta = words.to(torch.uint16).view(torch.float16)
    flat = torch.cat([values.reshape(-1), meta])
    tail = (-flat.numel()) % M
    if tail:
        flat = torch.cat([flat, torch.zeros(tail, dtype=flat.dtype)])
    return flat.reshape(M, -1)


def _reference(dense_a, dense_b, bias, alpha):
    out = dense_a.float() @ dense_b.float()
    if alpha is not None:
        out = out * float(alpha)
    if bias is not None:
        out = out + bias.float()[None, :]
    return out


@pytest.mark.parametrize("M, K, N", [(64, 128, 32), (32, 64, 16), (96, 128, 24)])
def test_cslt_sparse_mm_matches_formula(M, K, N):
    torch.manual_seed(0)
    dense_a = _make_2to4(M, K, torch.float16)
    compressed = _compress_2to4(dense_a)
    dense_b = torch.randn(K, N, dtype=torch.float16).clamp(-2, 2)

    out = _cslt_sparse_mm(compressed, dense_b)
    ref = _reference(dense_a, dense_b, None, None).to(torch.float16)
    assert out.shape == (M, N) and out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_cslt_sparse_mm_alpha_and_bias():
    torch.manual_seed(1)
    M, K, N = 64, 128, 32
    dense_a = _make_2to4(M, K, torch.float16)
    compressed = _compress_2to4(dense_a)
    dense_b = torch.randn(K, N, dtype=torch.float16).clamp(-2, 2)
    bias = torch.randn(N, dtype=torch.float16)

    out = _cslt_sparse_mm(compressed, dense_b, bias=bias, alpha=2.0)
    ref = _reference(dense_a, dense_b, bias, 2.0).to(torch.float16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_cslt_sparse_mm_transpose_result():
    torch.manual_seed(2)
    M, K, N = 64, 128, 32
    dense_a = _make_2to4(M, K, torch.float16)
    compressed = _compress_2to4(dense_a)
    dense_b = torch.randn(K, N, dtype=torch.float16).clamp(-2, 2)

    out = _cslt_sparse_mm(compressed, dense_b, transpose_result=True)
    ref = _reference(dense_a, dense_b, None, None).t().contiguous().to(torch.float16)
    assert out.shape == (N, M)
    assert out.is_contiguous()
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_cslt_sparse_mm_out_dtype():
    torch.manual_seed(3)
    M, K, N = 32, 64, 16
    dense_a = _make_2to4(M, K, torch.float16)
    compressed = _compress_2to4(dense_a)
    dense_b = torch.randn(K, N, dtype=torch.float16).clamp(-2, 2)

    out = _cslt_sparse_mm(compressed, dense_b, out_dtype=torch.float32)
    assert out.dtype == torch.float32
    ref = _reference(dense_a, dense_b, None, None)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_cslt_sparse_mm_rejects_non_2d_blob():
    torch.manual_seed(4)
    dense_a = _make_2to4(32, 64, torch.float16)
    compressed = _compress_2to4(dense_a).reshape(-1)
    dense_b = torch.randn(64, 16, dtype=torch.float16)
    with pytest.raises(
        RuntimeError, match="expected a 2-D cuSPARSELt compressed input"
    ):
        _cslt_sparse_mm(compressed, dense_b)


def test_cslt_sparse_mm_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems._cslt_sparse_mm")
    launches = []
    kernel = mod._decompress_kernel
    real_run = kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    dense_a = _make_2to4(64, 128, torch.float16)
    compressed = _compress_2to4(dense_a)
    dense_b = torch.randn(128, 32, dtype=torch.float16).clamp(-2, 2)
    out = _cslt_sparse_mm(compressed, dense_b)
    ref = _reference(dense_a, dense_b, None, None).to(torch.float16)
    assert len(launches) >= 1
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
