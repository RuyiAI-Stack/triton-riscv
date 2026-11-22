import importlib

import pytest
import torch

from .trilinear import _trilinear, _trilinear_out


def _ref_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim):
    # Upstream contract (per the ported file): insert a size-1 dim at each
    # position in expand1/2/3 (aten unsqueeze semantics), broadcast the three
    # operands to one shape, multiply elementwise, then sum over sumdim.
    a, b, c = i1.clone(), i2.clone(), i3.clone()
    for d in expand1:
        a = a.unsqueeze(d)
    for d in expand2:
        b = b.unsqueeze(d)
    for d in expand3:
        c = c.unsqueeze(d)
    shape = torch.broadcast_shapes(a.shape, b.shape, c.shape)
    prod = a.broadcast_to(shape) * b.broadcast_to(shape) * c.broadcast_to(shape)
    return prod.sum(dim=list(sumdim)) if len(sumdim) else prod


def test_trilinear_pure_product_no_sumdim():
    torch.manual_seed(0)
    i1 = torch.randn(2, 3, 4, dtype=torch.float32)
    i2 = torch.randn(2, 3, 4, dtype=torch.float32)
    i3 = torch.randn(2, 3, 4, dtype=torch.float32)
    out = _trilinear(i1, i2, i3, [], [], [], [])
    ref = _ref_trilinear(i1, i2, i3, [], [], [], [])
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_trilinear_expand_broadcast_product():
    torch.manual_seed(1)
    i1 = torch.randn(2, 3, dtype=torch.float32)
    i2 = torch.randn(3, 4, dtype=torch.float32)
    i3 = torch.randn(2, 3, 4, dtype=torch.float32)
    # a -> (2, 1, 3), b -> (1, 3, 4), c unchanged: full broadcast product.
    out = _trilinear(i1, i2, i3, [2], [0], [])
    ref = _ref_trilinear(i1, i2, i3, [2], [0], [], [])
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_trilinear_sumdim_matches_reference():
    torch.manual_seed(2)
    i1 = torch.randn(2, 3, dtype=torch.float32)
    i2 = torch.randn(3, 4, dtype=torch.float32)
    i3 = torch.randn(2, 3, 4, dtype=torch.float32)
    out = _trilinear(i1, i2, i3, [2], [0], [], [1])
    ref = _ref_trilinear(i1, i2, i3, [2], [0], [], [1])
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    out2 = _trilinear(i1, i2, i3, [2], [0], [], [1, 2])
    ref2 = _ref_trilinear(i1, i2, i3, [2], [0], [], [1, 2])
    assert out2.shape == (2,)
    torch.testing.assert_close(out2, ref2, rtol=1e-5, atol=1e-5)


def test_trilinear_classic_einsum_backward_shape():
    # einsum-backward flavor: every operand gets an inserted size-1 dim and
    # the contraction runs over that inserted dim. (The textbook aten shape
    # triple (E,1,m)/(1,E,n)/(E,m,n) is not mutually broadcastable in torch
    # itself -- dims 1 collide as 1/E/m -- so the equivalent broadcastable
    # geometry (E,m)/(m,n)/(E,m,n) with trailing inserts is pinned here.)
    torch.manual_seed(3)
    e_, m, n = 2, 3, 4
    i1 = torch.randn(e_, m, dtype=torch.float32)
    i2 = torch.randn(m, n, dtype=torch.float32)
    i3 = torch.randn(e_, m, n, dtype=torch.float32)
    out = _trilinear(i1, i2, i3, [2, 3], [0, 3], [3], [3])
    ref = _ref_trilinear(i1, i2, i3, [2, 3], [0, 3], [3], [3])
    assert out.shape == (e_, m, n)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_trilinear_unroll_dim_does_not_change_result():
    torch.manual_seed(4)
    i1 = torch.randn(2, 3, dtype=torch.float32)
    i2 = torch.randn(3, 4, dtype=torch.float32)
    i3 = torch.randn(2, 3, 4, dtype=torch.float32)
    out = _trilinear(i1, i2, i3, [2], [0], [], [2], unroll_dim=2)
    ref = _ref_trilinear(i1, i2, i3, [2], [0], [], [2])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_trilinear_arange_exact_float_dtypes():
    # Integer-valued data makes both the product and the reduction exact.
    i1 = torch.arange(2 * 3, dtype=torch.float64).reshape(2, 3) / 3
    i2 = torch.arange(3 * 4, dtype=torch.float64).reshape(3, 4) / 4
    i3 = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4) / 5
    torch.testing.assert_close(
        _trilinear(i1, i2, i3, [2], [0], [], [2]),
        _ref_trilinear(i1, i2, i3, [2], [0], [], [2]),
        rtol=0,
        atol=0,
    )
    f32 = [t.to(torch.float32) for t in (i1, i2, i3)]
    torch.testing.assert_close(
        _trilinear(*f32, [2], [0], [], [2]),
        _ref_trilinear(*f32, [2], [0], [], [2]),
        rtol=0,
        atol=0,
    )


def test_trilinear_fp64_randn():
    # Product is dtype-native; only the reduction order differs from the
    # .sum() reference. rtol/atol 1e-9 measured against the fp64 reference.
    torch.manual_seed(5)
    i1 = torch.randn(2, 3, dtype=torch.float64)
    i2 = torch.randn(3, 4, dtype=torch.float64)
    i3 = torch.randn(2, 3, 4, dtype=torch.float64)
    out = _trilinear(i1, i2, i3, [2], [0], [], [2])
    torch.testing.assert_close(
        out, _ref_trilinear(i1, i2, i3, [2], [0], [], [2]), rtol=1e-9, atol=1e-9
    )


def test_trilinear_int64_small_values_exact():
    i1 = torch.arange(2 * 3, dtype=torch.int64).reshape(2, 3) % 5
    i2 = torch.arange(3 * 4, dtype=torch.int64).reshape(3, 4) % 4
    i3 = torch.arange(2 * 3 * 4, dtype=torch.int64).reshape(2, 3, 4) % 3
    out = _trilinear(i1, i2, i3, [2], [0], [], [2])
    ref = _ref_trilinear(i1, i2, i3, [2], [0], [], [2])
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_trilinear_empty_input():
    torch.manual_seed(6)
    i1 = torch.empty(0, 3, dtype=torch.float32)
    i2 = torch.randn(3, 4, dtype=torch.float32)
    i3 = torch.empty(0, 3, 4, dtype=torch.float32)
    out = _trilinear(i1, i2, i3, [2], [0], [], [])
    ref = _ref_trilinear(i1, i2, i3, [2], [0], [], [])
    assert out.shape == (0, 3, 4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_trilinear_sum_over_empty_dim_is_zero():
    torch.manual_seed(7)
    i1 = torch.empty(0, 3, dtype=torch.float32)
    i2 = torch.randn(3, 4, dtype=torch.float32)
    i3 = torch.empty(0, 3, 4, dtype=torch.float32)
    out = _trilinear(i1, i2, i3, [2], [0], [], [0])
    ref = _ref_trilinear(i1, i2, i3, [2], [0], [], [0])
    assert out.shape == (3, 4)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_trilinear_non_contiguous_input():
    torch.manual_seed(8)
    base = torch.randn(4, 2, dtype=torch.float32)
    i1 = base.t()  # (2, 4) non-contiguous view
    assert not i1.is_contiguous()
    i2 = torch.randn(4, 3, dtype=torch.float32)
    i3 = torch.randn(2, 4, 3, dtype=torch.float32)
    out = _trilinear(i1, i2, i3, [2], [0], [], [2])
    ref = _ref_trilinear(i1, i2, i3, [2], [0], [], [2])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_trilinear_error_matches_upstream_validation():
    torch.manual_seed(9)
    i2 = torch.randn(2, 3, dtype=torch.float32)
    i3 = torch.randn(2, 3, dtype=torch.float32)
    with pytest.raises(TypeError, match="_trilinear expects three tensor inputs"):
        _trilinear([1.0, 2.0], i2, i3, [], [], [], [])


def test_trilinear_out_writes_buffer():
    torch.manual_seed(10)
    i1 = torch.randn(2, 3, dtype=torch.float32)
    i2 = torch.randn(3, 4, dtype=torch.float32)
    i3 = torch.randn(2, 3, 4, dtype=torch.float32)
    out = torch.empty(2, 4, dtype=torch.float32)
    ret = _trilinear_out(i1, i2, i3, [2], [0], [], [2], out=out)
    assert ret is out
    torch.testing.assert_close(
        out, _ref_trilinear(i1, i2, i3, [2], [0], [], [2]), rtol=1e-5, atol=1e-5
    )


def test_trilinear_triton_path(monkeypatch):
    mod = importlib.import_module("flaggems.trilinear")
    launches = []
    kernel = mod.trilinear_kernel  # kernel name per upstream
    real_run = kernel.run

    def spy_run(*args, _real=real_run, **kwargs):
        launches.append(kwargs.get("grid"))
        return _real(*args, **kwargs)

    monkeypatch.setattr(kernel, "run", spy_run)
    torch.manual_seed(11)
    i1 = torch.randn(2, 3, dtype=torch.float64)
    i2 = torch.randn(3, 4, dtype=torch.float64)
    i3 = torch.randn(2, 3, 4, dtype=torch.float64)
    out = _trilinear(i1, i2, i3, [2], [0], [], [2])
    assert len(launches) >= 1
    torch.testing.assert_close(
        out, _ref_trilinear(i1, i2, i3, [2], [0], [], [2]), rtol=1e-9, atol=1e-9
    )
