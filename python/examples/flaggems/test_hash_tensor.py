import importlib
import struct

import pytest
import torch

from .hash_tensor import hash_tensor

# torch reference: aten::hash_tensor is registered and CPU-callable on this
# build (the underscored torch._hash_tensor spelling is not).
torch_hash = torch.ops.aten.hash_tensor

FLOAT_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
INT_DTYPES = [
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]

SHAPES = [(64,), (1024,), (7, 33), (16, 32, 64), (2, 3, 4, 5)]


def assert_hash_equal(res, ref):
    # Bit-exact comparison; the int64 view keeps it independent of any
    # uint64 comparison quirks (mirrors the upstream test helper).
    assert res.dtype == torch.uint64
    assert res.shape == ref.shape
    assert torch.equal(res.view(torch.int64), ref.view(torch.int64))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor_float(shape, dtype, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(dtype)
    dim_arg = [] if dim is None else [dim]

    ref = torch_hash(x, dim_arg, keepdim=keepdim, mode=0)
    res = hash_tensor(x, dim_arg, keepdim=keepdim, mode=0)

    assert_hash_equal(res, ref)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("dim", [None, -1])
def test_hash_tensor_int(shape, dtype, dim):
    torch.manual_seed(0)
    if dtype == torch.bool:
        x = torch.randint(0, 2, shape).to(dtype)
    else:
        x = torch.randint(-1000, 1000, shape).to(dtype)
    dim_arg = [] if dim is None else [dim]

    ref = torch_hash(x, dim_arg, keepdim=False, mode=0)
    res = hash_tensor(x, dim_arg, keepdim=False, mode=0)

    assert_hash_equal(res, ref)


@pytest.mark.parametrize(
    "shape,dim",
    [
        # Multi-axis reductions: trailing-contiguous, leading, and a
        # non-collapsible mix that forces the wrapper's materializing path.
        ((64, 128, 256), [1, 2]),
        ((16, 32, 64), [0, 1]),
        ((8, 16, 32, 64), [0, 2]),
        ((8, 16, 32, 64), [-1, -3]),  # negative multi-dim
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor_multidim(shape, dim, keepdim):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)

    ref = torch_hash(x, dim, keepdim=keepdim, mode=0)
    res = hash_tensor(x, dim, keepdim=keepdim, mode=0)

    assert_hash_equal(res, ref)


@pytest.mark.parametrize(
    "value, dtype, bits_fn",
    [
        (42, torch.int64, lambda v: struct.unpack("<Q", struct.pack("<q", v))[0]),
        (3.140000104904175, torch.float32, None),  # float32(3.14) exact value
        (1.5, torch.float64, lambda v: struct.unpack("<Q", struct.pack("<d", v))[0]),
    ],
)
def test_hash_tensor_scalar(value, dtype, bits_fn):
    x = torch.tensor(value, dtype=dtype)

    ref = torch_hash(x, [], keepdim=False, mode=0)
    res = hash_tensor(x, [], keepdim=False, mode=0)

    assert res.shape == ()
    assert_hash_equal(res, ref)
    if bits_fn is not None:
        assert res.item() == bits_fn(value)
    else:
        assert (
            res.item()
            == struct.unpack(
                "<Q", struct.pack("<d", float(torch.tensor(value, dtype=torch.float32)))
            )[0]
        )


def test_hash_tensor_known_bit_patterns():
    # Independent numeric reference, computed by hand from the widening
    # contract: floats -> float64 bits, ints -> sign-extended int64 bits.
    x = torch.tensor([1.5, -2.25, 0.0, 3.0], dtype=torch.float32)
    manual = 0
    for v in x.tolist():
        manual ^= struct.unpack("<Q", struct.pack("<d", float(v)))[0]
    assert hash_tensor(x, []).item() == manual

    xi = torch.tensor([-3, 7, 100, -100], dtype=torch.int8)
    manual = 0
    for v in xi.tolist():
        manual ^= struct.unpack("<Q", struct.pack("<q", int(v)))[0]
    assert hash_tensor(xi, []).item() == manual

    xu = torch.tensor([3, 200, 1, 0], dtype=torch.uint8)
    assert hash_tensor(xu, []).item() == (3 ^ 200 ^ 1 ^ 0)


def test_hash_tensor_int_dim_argument():
    # A bare int dim is accepted and equals the single-element list form.
    torch.manual_seed(0)
    x = torch.randn(6, 40, dtype=torch.float32)

    ref = torch_hash(x, [1], keepdim=False, mode=0)
    res = hash_tensor(x, 1, keepdim=False, mode=0)

    assert_hash_equal(res, ref)


def test_hash_tensor_non_contiguous():
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=torch.float32).t()
    assert not x.is_contiguous()

    for dim_arg in ([0], [1], []):
        ref = torch_hash(x, dim_arg, keepdim=False, mode=0)
        res = hash_tensor(x, dim_arg, keepdim=False, mode=0)
        assert_hash_equal(res, ref)


def test_hash_tensor_empty_output_side():
    # M == 0 (all rows reduced away by a zero-size kept dim) is defined by
    # aten: empty output, no error.
    x = torch.zeros(0, 5, dtype=torch.float32)
    ref = torch_hash(x, [1], keepdim=False, mode=0)
    res = hash_tensor(x, [1], keepdim=False, mode=0)
    assert_hash_equal(res, ref)
    assert res.numel() == 0


def test_hash_tensor_empty_reduce_side_upstream_contract():
    # Divergence pin (upstream flag_gems semantics): a zero-size reduction
    # dim yields zero hashes, while aten raises
    #   IndexError: hash_tensor: Expected reduction dim 1 to have non-zero
    #   size.
    # The port follows the upstream module, so this case is asserted
    # against the contract instead of the aten oracle.
    x = torch.zeros(3, 0, dtype=torch.float32)
    with pytest.raises(IndexError):
        torch_hash(x, [1], keepdim=False, mode=0)

    res = hash_tensor(x, [1], keepdim=False, mode=0)
    assert res.shape == (3,)
    assert (res == 0).all()


def test_hash_tensor_full_empty_upstream_contract():
    # Full reduction of a numel == 0 tensor: aten raises RuntimeError
    # ("Expected reduction dim to be specified for input.numel() == 0");
    # the upstream module returns a zero scalar. Port pins the upstream
    # contract.
    x = torch.zeros(0, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch_hash(x, [], keepdim=False, mode=0)

    res = hash_tensor(x, [], keepdim=False, mode=0)
    assert res.shape == ()
    assert res.item() == 0


def test_hash_tensor_duplicate_dims_upstream_contract():
    # aten rejects repeated dims ("dim 0 appears multiple times"); upstream
    # dedups, so [0, 0] == [0].
    x = torch.randn(4, 8, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch_hash(x, [0, 0], keepdim=False, mode=0)

    assert_hash_equal(hash_tensor(x, [0, 0]), hash_tensor(x, [0]))


def test_hash_tensor_rejects_unknown_mode():
    x = torch.randn(8, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Unknown hash_tensor mode"):
        hash_tensor(x, [], mode=1)
    with pytest.raises(RuntimeError):
        torch_hash(x, [], mode=1)


def test_hash_tensor_triton_path(monkeypatch):
    # Smoke: the main path must launch exactly one Triton program per
    # output row.
    mod = importlib.import_module("flaggems.hash_tensor")

    launches = []
    real_run = mod._hash_row_kernel.run

    def spy_run(*args, **kwargs):
        launches.append(kwargs.get("grid"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(mod._hash_row_kernel, "run", spy_run)
    torch.manual_seed(0)
    x = torch.randn(12, 40, dtype=torch.float32)
    res = mod.hash_tensor(x, [1], keepdim=False, mode=0)
    assert launches == [(12,)]
    assert_hash_equal(res, torch_hash(x, [1], keepdim=False, mode=0))
